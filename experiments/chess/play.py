"""Run in PyCharm, then open http://127.0.0.1:8000 to play the saved policy.

Uses only the training dependencies (torch, numpy, chess) and Python's standard
library. No CDN, web framework, or internet connection is needed. One local game
is shared by all browser tabs. Edit SETTINGS below; no command-line arguments.
For optional Stockfish matches, set MODE = "benchmark" and STOCKFISH_PATH.
"""
from __future__ import annotations

import json
import math
import shutil
import webbrowser
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import chess.svg
import numpy as np
import torch

from main import (ChessPolicy, Config, action_index, encode_batch, game_result,
                  legal_moves, read_checkpoint, plot_history, checkpoint_path,
                  ARCHITECTURE_KEYS, LEGACY_MODEL_VERSION)

# SETTINGS
DIRECTORY = Path(__file__).resolve().parent
CHECKPOINT_PATH = checkpoint_path(Config())
if not CHECKPOINT_PATH.is_file():
    for name in (f"chess_pretrained_d{Config().dim}.pt", "chess_pretrained.pt"):
        candidate = CHECKPOINT_PATH.parent / name
        if candidate.is_file():
            CHECKPOINT_PATH = candidate
            break
MODE = "play"  # "play", "training" (human-game training charts), or "benchmark" (Stockfish)
DEVICE = "auto"
HOST = "127.0.0.1"
PORT = 8000
OPEN_BROWSER = True
SEED = 42
DEFAULT_SAMPLING = {"temperature": 1.0, "min_p": 0.1}
ENDGAMES = {
    "King + queen": "7k/8/5K2/8/8/8/8/1Q6 w - - 0 1",
    "King + rook": "7k/8/5K2/8/8/8/8/R7 w - - 0 1",
    "King + rook + knight": "7k/8/5K2/8/8/8/8/RN6 w - - 0 1",
}
# Temperature applies to dimensionless actor logits. Zero means greedy.
STOCKFISH_PATH = shutil.which("stockfish") or "/usr/games/stockfish"
STOCKFISH_ELOS = (1320, 1600, 2000)  # Must fit the installed engine's UCI_Elo range.
GAMES_PER_ELO = 20  # Even: paired openings with colours swapped.
STOCKFISH_SECONDS = 0.1  # Fixed engine time per move, not an official rating time control.
MAX_PLIES = 400  # Capped games are unfinished, never counted as draws.
BENCHMARK_SAMPLING = {**DEFAULT_SAMPLING}
RESULTS_DIRECTORY = DIRECTORY.parents[1] / "data" / "chess" / "evaluation"
OPENINGS = ((), ("e2e4", "e7e5"), ("d2d4", "d7d5"), ("c2c4", "e7e5"))


def load_model():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Set CHECKPOINT_PATH to a trained model: {CHECKPOINT_PATH}")
    # Rebuild the saved architecture, regardless of today's training Config.
    saved = read_checkpoint(CHECKPOINT_PATH, "cpu")
    architecture = {key: saved["architecture"][key] for key in ARCHITECTURE_KEYS}
    legacy = saved["model_version"] == LEGACY_MODEL_VERSION
    legacy_blocks = architecture.pop("blocks") if legacy else None
    cfg = replace(Config(), **architecture, rules=saved["rules"],
                  temperature=saved.get("policy_temperature", 1.))
    name = DEVICE
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(name)
    network = ChessPolicy(cfg, legacy_blocks=legacy_blocks).to(device)
    network.load_state_dict(saved["ema_model"] if network.has_value else saved["model"])
    network.eval()
    network.pretraining_summary = saved.get("pretraining")
    network.training_progress = saved.get("training_progress")
    print(f"Loaded {CHECKPOINT_PATH} | {training_summary(network)} | "
          f"{cfg.rules} | {device}", flush=True)
    return network, cfg, device


def sampling_options(raw):
    temperature = float(raw.get("temperature", DEFAULT_SAMPLING["temperature"]))
    min_p = float(raw.get("min_p", DEFAULT_SAMPLING["min_p"]))
    if not math.isfinite(temperature) or not 0 <= temperature <= 100:
        raise ValueError("Temperature must be between 0 and 100.")
    if not math.isfinite(min_p) or not 0 <= min_p <= 1:
        raise ValueError("Min-p must be between 0 and 1.")
    return dict(temperature=temperature, min_p=min_p)


def policy_logits(values):
    scores = np.asarray(values, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("Model produced non-finite scores.")
    return scores


def probabilities(values, options):
    """Apply temperature, retain p >= min_p * p_max, then renormalize."""
    scores = policy_logits(values)
    probs = np.zeros(len(scores), dtype=np.float64)
    if not len(scores):
        return probs
    if options["temperature"] == 0:
        probs[int(np.argmax(scores))] = 1
        return probs
    logits = (scores - scores.max()) / options["temperature"]
    weights = np.exp(logits)
    # max(weights) == 1; probability ratios equal these unnormalized weights.
    weights[weights < options.get("min_p", DEFAULT_SAMPLING["min_p"])] = 0
    return weights / weights.sum()


class Player:
    def __init__(self, network, cfg, device):
        self.network, self.cfg, self.device = network, cfg, device
        self.board = chess.Board()
        self.human = None  # None lets the user move either side, on its turn.
        self.rng = np.random.default_rng(SEED)
        self.cached_fen = None
        self.cached_analysis = None
        self.last_model = ""

    def evaluate(self, options=None):
        moves, logits, _ = self.current_analysis()
        return moves, logits

    def current_analysis(self):
        key = (self.board.fen(), tuple(self.board.move_stack))
        if key != self.cached_fen:
            self.cached_analysis = self.evaluate_position(self.board)
            self.cached_fen = key
        return self.cached_analysis

    @torch.inference_mode()
    def evaluate_position(self, board):
        ended, winner = game_result(board, self.cfg.rules)
        if ended:
            return [], np.array([]), .5 if winner is None else float(winner == chess.WHITE)
        moves = legal_moves(board, self.cfg.rules)
        inputs = encode_batch([board], self.device)
        if self.network.has_value:
            logits, value = self.network(*inputs, with_value=True)
            white_score = value.sigmoid().item()
        else:
            logits, white_score = self.network(*inputs), None
        indices = [action_index(move) for move in moves]
        return moves, logits[0, indices].cpu().numpy(), white_score

    def model_move(self, options):
        moves, values = self.evaluate()
        if not moves:
            raise ValueError("The game has finished.")
        probs = probabilities(values, options)
        i = int(self.rng.choice(len(probs), p=probs))
        move = moves[i]
        self.last_model = (f"{self.board.san(move)} · logit {policy_logits(values)[i]:+.3f} · "
                           f"sample probability {probs[i]:.1%}")
        self.board.push(move)

    def state(self, options):
        moves, values, white_score = self.current_analysis()
        probs = probabilities(values, options)
        scores = policy_logits(values)
        order = np.argsort(-scores, kind="stable")
        rows = []
        for i in order[:12]:
            move = moves[i]
            rows.append(dict(move=self.board.san(move), logit=float(scores[i]), p=float(probs[i])))
        ended, winner = game_result(self.board, self.cfg.rules)
        status = ("Draw" if winner is None else ("White" if winner else "Black") + " wins") if ended else (
                ("White" if self.board.turn else "Black") + " to move")
        if self.cfg.rules == "standard":
            outcome = self.board.outcome(claim_draw=True)
            if outcome:
                status += " · " + outcome.termination.name.replace("_", " ").lower()
            elif self.board.is_check():
                status += " · check"
        game = chess.pgn.Game.from_board(self.board)
        if ended:
            game.headers["Result"] = "1/2-1/2" if winner is None else "1-0" if winner else "0-1"
        return dict(fen=self.board.fen(), pieces={chess.square_name(s): p.symbol()
                                                  for s, p in self.board.piece_map().items()},
                    moves=[] if ended else [m.uci() for m in legal_moves(self.board, self.cfg.rules)],
                    last=self.board.peek().uci() if self.board.move_stack else "",
                    status=status, ended=ended, turn=self.board.turn, human=self.human,
                    rows=rows, last_model=self.last_model, pgn=str(game),
                    history=game.accept(chess.pgn.StringExporter(headers=False, comments=False, variations=False)),
                    plies=len(self.board.move_stack), rules=self.cfg.rules,
                    training=training_summary(self.network), white_score=white_score)

    def action(self, path, body):
        options = sampling_options(body.get("sampling", {}))
        if path in ("/new", "/control"):
            side = body.get("side", "both")
            if side not in ("white", "black", "both"):
                raise ValueError("Choose control of White, Black, or both sides.")
            self.human = None if side == "both" else side == "white"
        if path == "/new":
            self.board = chess.Board()
            self.last_model = ""
        elif path == "/position":
            board = chess.Board(body["fen"].strip())
            if self.cfg.rules == "standard" and not board.is_valid():
                raise ValueError("Invalid standard chess position.")
            self.board = board
            self.last_model = ""
        elif path == "/undo":
            if self.board.move_stack:
                self.board.pop()
            self.last_model = ""
        elif path == "/move":
            if self.human is not None and self.board.turn != self.human:
                raise ValueError("It is the model's turn.")
            move = chess.Move.from_uci(body["move"])
            if game_result(self.board, self.cfg.rules)[0] or move not in legal_moves(self.board, self.cfg.rules):
                raise ValueError("That move is not legal.")
            self.board.push(move)
        elif path == "/model":
            self.model_move(options)
        elif path not in ("/state", "/control"):
            raise ValueError("Unknown action.")
        return self.state(options)


PAGE = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Chess policy · Play the checkpoint</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#121a21;color:#e7ede9;font:15px system-ui,sans-serif}
main{max-width:1150px;margin:auto;padding:28px}h1{margin:0;font-size:28px}p{color:#a8b8bc;line-height:1.5}
.layout{display:grid;grid-template-columns:minmax(280px,610px) minmax(270px,1fr);gap:28px;margin-top:24px}
#board{display:grid;grid-template-columns:repeat(8,minmax(0,1fr));grid-template-rows:repeat(8,minmax(0,1fr));width:100%;aspect-ratio:1;touch-action:none;user-select:none}
.square{position:relative;display:flex;align-items:center;justify-content:center;min-width:0;min-height:0;background:#e1dbc8;color:#283d36}
.square.dark{background:#698b7c}.square img{position:absolute;inset:4%;width:92%;height:92%;object-fit:contain;pointer-events:none}.square.selected{box-shadow:inset 0 0 0 5px #e4b853}
.square.last{background:#c6c68a}.square.last.dark{background:#a1ad63}.square.target:after{content:'';width:24%;height:24%;border-radius:50%;background:#233b4477;position:absolute}
.coord{position:absolute;bottom:2px;left:4px;font-size:10px;pointer-events:none}
#ghost{position:fixed;pointer-events:none;z-index:5;display:none;transform:translate(-50%,-50%);width:65px;height:65px}
.panel{background:#1c2831;padding:20px;border-radius:12px}label{display:block;margin:15px 0 6px}input[type=range]{width:100%;accent-color:#8bd4b1}
button,select,input[type=text]{background:#2a3c47;color:#eef4ef;border:1px solid #4c646b;padding:9px;border-radius:6px;font:inherit}
button{cursor:pointer}button:disabled{opacity:.4;cursor:wait}.buttons{display:flex;gap:8px;flex-wrap:wrap;margin:12px 0}
input[type=text]{width:100%}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:5px;border-bottom:1px solid #35454e}th:first-child,td:first-child{text-align:left}
#evalBar{height:12px;background:#151b20;border:1px solid #667781;border-radius:6px;overflow:hidden;margin-top:8px}
#evalFill{height:100%;background:#f0eee5}
#evalText{margin:4px 0 0}
#status{font-size:20px;margin:14px 0}#error{color:#ffb5a6;min-height:24px}.small{font-size:12px;color:#a8b8bc}
#history{background:#1c2831;padding:12px;border-radius:8px;min-height:3em;max-height:160px;overflow:auto;white-space:pre-wrap;font:inherit;line-height:1.6}
@media(max-width:760px){main{padding:16px}.layout{grid-template-columns:1fr}}
</style>
<main><h1>Chess policy <span class="small">/ play the checkpoint</span></h1>
<p id="training" class="small"></p>
<p>Drag a piece or click its square, then its destination.</p>
<div class="layout"><section><div id="board"></div><div id="evalBar" role="meter" aria-label="White expected score" aria-valuemin="0" aria-valuemax="100" hidden><div id="evalFill"></div></div><p id="evalText" class="small"></p><div id="status"></div><div id="error" role="alert"></div>
<strong>Game history</strong><pre id="history"></pre>
<div class="buttons"><select id="side" aria-label="Sides to control"><option value="both">Control both sides</option><option value="white">Play as White</option><option value="black">Play as Black</option></select><button id="new">New game</button><button id="undo">Undo ply</button><button id="flip">Flip board</button><button id="download">Save PGN</button></div>
<label>Position (FEN)</label><input id="fen" type="text"><div class="buttons"><button id="load">Load position</button></div>
<label>Endgame examples</label><select id="endgame"><option value="">Choose a position</option></select>
<p class="small">Paste any valid FEN to configure pieces and the side to move. Loading a position resets move history.</p>
</section>
<section class="panel"><strong>Move sampling</strong>
<label for="temperature">Temperature</label><input id="temperature" type="number" min="0" max="100" step="any">
<label for="min_p">Min-p (0–1)</label><input id="min_p" type="number" min="0" max="1" step="any">
<p class="small">The model samples from legal moves on the current board. Min-p keeps moves at least this fraction as likely as the favourite.</p>
<label><input id="auto" type="checkbox" checked> Model replies automatically</label>
<div class="buttons"><button id="model">Make model move</button><button id="refresh">Refresh scores</button></div>
<p id="modelInfo" class="small"></p><strong>Top legal moves</strong>
<table><thead><tr><th>Move</th><th>Policy logit</th><th>P (filtered)</th></tr></thead><tbody id="scores"></tbody></table>
<p class="small" id="details"></p><p class="small">Logits are shown before temperature and min-p. Draws are claimed automatically when available. FEN loads start a new repetition history.</p></section></div></main><img id="ghost" alt="">
<script>
const icons=__ICONS__;
const $=id=>document.getElementById(id);
let state=null,flipped=false,selected=null,busy=false,drag=null;
const fields=['temperature','min_p'];
function sampling(){return Object.fromEntries(fields.map(id=>[id,$(id).valueAsNumber]));}
function render(){
 if(!state)return;
 const board=$('board');board.replaceChildren();
 const files=flipped?'hgfedcba':'abcdefgh',ranks=flipped?'12345678':'87654321';
 for(const r of ranks)for(const f of files){
  const sq=f+r,el=document.createElement('div');el.dataset.square=sq;el.className='square';
  if((f.charCodeAt(0)-97+(+r)-1)%2===0)el.classList.add('dark');
  if(state.last.slice(0,2)===sq||state.last.slice(2,4)===sq)el.classList.add('last');
  if(sq===selected)el.classList.add('selected');
  if(selected&&state.moves.some(m=>m.slice(0,2)===selected&&m.slice(2,4)===sq))el.classList.add('target');
  if(state.pieces[sq]){const img=document.createElement('img');img.src=icons[state.pieces[sq]];img.alt=state.pieces[sq];el.append(img);}
  const coord=document.createElement('span');coord.className='coord';coord.textContent=sq;el.append(coord);board.append(el);
 }
 $('status').textContent=state.status;$('fen').value=state.fen;$('modelInfo').textContent=state.last_model;
 $('training').textContent=state.training;
 $('side').value=state.human===null?'both':state.human?'white':'black';
 $('auto').disabled=state.human===null;
 $('evalBar').hidden=state.white_score===null;
 if(state.white_score!==null){
  const percent=(100*state.white_score).toFixed(1);
  $('evalFill').style.width=percent+'%';
  $('evalBar').setAttribute('aria-valuenow',percent);
  $('evalText').textContent=`${state.ended?'Final White score':'White expected score'} ${percent}%`;
 }else $('evalText').textContent='';
 $('details').textContent=`${state.rules} · ${state.plies} half-moves played · automatic queen promotion`;
 $('history').textContent=state.plies?state.history:'No moves yet';
 $('history').scrollTop=$('history').scrollHeight;
 $('scores').replaceChildren();
 for(const row of state.rows){const tr=document.createElement('tr');for(const text of [row.move,row.logit.toFixed(3),(100*row.p).toFixed(1)+'%']){const td=document.createElement('td');td.textContent=text;tr.append(td)}$('scores').append(tr)}
}
async function request(path,extra={}){
 const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sampling:sampling(),...extra})});
 const data=await response.json();if(!response.ok)throw Error(data.error);state=data;selected=null;render();
}
async function run(path,extra={},reply=false){
 if(busy)return;busy=true;$('error').textContent='';document.querySelectorAll('button').forEach(b=>b.disabled=true);
 try{await request(path,extra);if(reply&&$('auto').checked&&state.human!==null&&!state.ended&&state.turn!==state.human)await request('/model');}
 catch(e){$('error').textContent=e.message}
 finally{busy=false;document.querySelectorAll('button').forEach(b=>b.disabled=false)}
}
function movable(sq){return state&&!state.ended&&(state.human===null||state.turn===state.human)&&state.moves.some(m=>m.slice(0,2)===sq)}
function destination(sq){
 if(!selected)return false;const choices=state.moves.filter(m=>m.slice(0,2)===selected&&m.slice(2,4)===sq);
 if(!choices.length)return false;run('/move',{move:choices[0]},true);return true;
}
$('board').onpointerdown=e=>{
 if(busy||e.button!==0)return;const sq=e.target.closest('[data-square]')?.dataset.square;if(!sq)return;
 if(selected&&sq!==selected&&destination(sq))return;
 if(!movable(sq)){selected=null;render();return;}
 selected=sq;drag={sq,x:e.clientX,y:e.clientY,moved:false};render();
};
window.addEventListener('pointermove',e=>{
 if(!drag)return;if(Math.hypot(e.clientX-drag.x,e.clientY-drag.y)>5)drag.moved=true;
 if(drag.moved){const ghost=$('ghost');ghost.src=icons[state.pieces[drag.sq]];ghost.style.display='block';ghost.style.left=e.clientX+'px';ghost.style.top=e.clientY+'px';}
});
window.addEventListener('pointerup',e=>{
 if(!drag)return;const moved=drag.moved;drag=null;$('ghost').style.display='none';
 if(moved){const sq=document.elementFromPoint(e.clientX,e.clientY)?.closest('[data-square]')?.dataset.square;if(sq)destination(sq);}
});
window.addEventListener('pointercancel',()=>{drag=null;$('ghost').style.display='none'});
$('new').onclick=()=>{flipped=$('side').value==='black';run('/new',{side:$('side').value},true)};
$('side').onchange=()=>run('/control',{side:$('side').value},true);
$('undo').onclick=()=>run('/undo');$('flip').onclick=()=>{flipped=!flipped;render()};
$('model').onclick=()=>run('/model');$('refresh').onclick=()=>run('/state');
$('load').onclick=()=>run('/position',{fen:$('fen').value},true);
$('auto').onchange=()=>{if($('auto').checked&&state&&state.human!==null&&!state.ended&&state.turn!==state.human)run('/model')};
for(const id of fields)$(id).onchange=()=>run('/state');
const endgames=__ENDGAMES__;
for(const [name,fen] of Object.entries(endgames)){const option=document.createElement('option');option.value=fen;option.textContent=name;$('endgame').append(option)}
$('endgame').onchange=()=>{const fen=$('endgame').value;$('endgame').value='';if(fen)run('/position',{fen})};
$('download').onclick=()=>{const url=URL.createObjectURL(new Blob([state.pgn+'\n'],{type:'application/x-chess-pgn'}));const a=document.createElement('a');a.href=url;a.download='chessq.pgn';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
const defaults=__SAMPLING__;
for(const id of fields)$(id).value=defaults[id];
run('/state');
</script></html>'''


def training_summary(network):
    summary = getattr(network, "pretraining_summary", None) or {}
    rows = summary.get("history", [])
    progress = getattr(network, "training_progress", None)
    if progress:
        previews = summary.get("previews", [])
        accuracy = f" · preview accuracy {previews[-1]['validation']['accuracy']:.1%}" if previews else ""
        return f"Human-game policy · epoch {progress['epoch']}, position {progress['positions']:,}{accuracy}"
    if not rows:
        return "Human-game policy · no saved epoch metrics"
    last = rows[-1]
    return (f"Human-game policy · epoch {last['epoch']} · held-out move accuracy "
            f"{last['validation']['accuracy']:.1%}")


def show_training(network, cfg):
    print(training_summary(network))
    summary = getattr(network, "pretraining_summary", None) or {}
    if summary.get("previews"):
        plot_history(summary["previews"], positions=True)
    if summary.get("history"):
        plot_history(summary["history"])


def serve(player):
    import base64
    icons = {s: "data:image/svg+xml;base64," + base64.b64encode(
        chess.svg.piece(chess.Piece.from_symbol(s)).encode()).decode()
             for s in "PNBRQKpnbrqk"}
    page = (PAGE.replace("__ICONS__", json.dumps(icons))
            .replace("__ENDGAMES__", json.dumps(ENDGAMES))
            .replace("__SAMPLING__", json.dumps(sampling_options(DEFAULT_SAMPLING))).encode())

    class Handler(BaseHTTPRequestHandler):
        def respond(self, code, content, content_type):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            if self.path != "/":
                self.respond(404, b"Not found", "text/plain")
                return
            self.respond(200, page, "text/html; charset=utf-8")

        def do_POST(self):
            try:
                # Mutations are accepted only from this local page / same origin.
                if self.headers.get("Origin") not in (None, f"http://{HOST}:{PORT}"):
                    raise ValueError("Unexpected request origin.")
                if self.headers.get("Content-Type") != "application/json":
                    raise ValueError("Expected JSON.")
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 16384:
                    raise ValueError("Invalid request length.")
                body = json.loads(self.rfile.read(length))
                state = player.action(self.path, body)
                self.respond(200, json.dumps(state, allow_nan=False).encode(), "application/json")
            except (ValueError, KeyError, TypeError) as exc:
                self.respond(400, json.dumps({"error": str(exc)}).encode(), "application/json")
            except Exception as exc:
                print(f"Inference failed: {exc}", flush=True)
                self.respond(500, json.dumps({"error": str(exc)}).encode(), "application/json")

        def log_message(self, *_):
            pass

    with HTTPServer((HOST, PORT), Handler) as server:
        url = f"http://{HOST}:{PORT}"
        print(f"Play at {url} · Ctrl+C to stop. One shared local game.", flush=True)
        if OPEN_BROWSER:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Stopped.")


def benchmark(player):
    """Paired games against limited-strength Stockfish; results aren't human Elo."""
    if player.cfg.rules != "standard":
        raise ValueError("Stockfish benchmarking requires a standard-rules checkpoint.")
    if GAMES_PER_ELO < 2 or GAMES_PER_ELO % 2:
        raise ValueError("Use a positive even GAMES_PER_ELO for colour-swapped pairs.")
    if not Path(STOCKFISH_PATH).is_file():
        raise FileNotFoundError("Install Stockfish separately and set STOCKFISH_PATH.")
    options = sampling_options(BENCHMARK_SAMPLING)
    summaries = []
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    output = RESULTS_DIRECTORY / f"matches_{datetime.now():%Y%m%d_%H%M%S_%f}.pgn"
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine, output.open("w") as pgn:
        elo_option = engine.options.get("UCI_Elo")
        if elo_option is None or "UCI_LimitStrength" not in engine.options:
            raise ValueError("This engine does not expose UCI_Elo / UCI_LimitStrength.")
        if any(not elo_option.min <= elo <= elo_option.max for elo in STOCKFISH_ELOS):
            raise ValueError(f"Engine UCI_Elo range: {elo_option.min}..{elo_option.max}; "
                             "edit STOCKFISH_ELOS to match.")
        engine.configure({"UCI_LimitStrength": True, "Threads": 1, "Hash": 64})
        print(f"Engine: {engine.id}; settings: {options}; {STOCKFISH_SECONDS}s/engine move")
        for elo in STOCKFISH_ELOS:
            engine.configure({"UCI_Elo": elo})
            counts = dict(wins=0, draws=0, losses=0, unfinished=0)
            for index in range(GAMES_PER_ELO):
                player.board = chess.Board()
                opening = OPENINGS[(index // 2) % len(OPENINGS)]
                for uci in opening:
                    player.board.push_uci(uci)
                model_colour = chess.WHITE if index % 2 == 0 else chess.BLACK
                game_token = object()  # python-chess sends ucinewgame once per match.
                for _ in range(MAX_PLIES):
                    if game_result(player.board, "standard")[0]:
                        break
                    if player.board.turn == model_colour:
                        player.model_move(options)
                    else:
                        result = engine.play(player.board, chess.engine.Limit(time=STOCKFISH_SECONDS),
                                             game=game_token)
                        player.board.push(result.move)
                outcome = player.board.outcome(claim_draw=True)
                key = "unfinished" if outcome is None else "draws" if outcome.winner is None else (
                    "wins" if outcome.winner == model_colour else "losses")
                counts[key] += 1
                game = chess.pgn.Game.from_board(player.board)
                game.headers.update(Event=f"Chess policy vs Stockfish UCI_Elo {elo}",
                                    White="ChessPolicy" if model_colour else engine.id.get("name", "Stockfish"),
                                    Black=engine.id.get("name", "Stockfish") if model_colour else "ChessPolicy",
                                    Result=outcome.result() if outcome else "*",
                                    Termination="normal" if outcome else "ply cap",
                                    Sampling=json.dumps(options), EngineSeconds=str(STOCKFISH_SECONDS))
                print(game, file=pgn, end="\n\n")
                pgn.flush()
                print(f"UCI_Elo {elo} | {index + 1}/{GAMES_PER_ELO} | {key} | {counts}", flush=True)
            summary = dict(stockfish_elo=elo, **counts)
            # Only report a score if every match finished: no silent truncation bias.
            if not counts["unfinished"]:
                score = (counts["wins"] + 0.5 * counts["draws"]) / GAMES_PER_ELO
                summary["score"] = score
                if 0 < score < 1:
                    summary["rough_engine_relative_elo"] = elo + 400 * math.log10(score / (1 - score))
            summaries.append(summary)
        metadata = dict(checkpoint=str(CHECKPOINT_PATH), engine=engine.id,
                        sampling=options, engine_seconds=STOCKFISH_SECONDS,
                        games_per_elo=GAMES_PER_ELO, max_plies=MAX_PLIES,
                        seed=SEED, results=summaries,
                        caveat="Engine-relative estimates only; small samples, openings and time settings matter.")
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))
    print(f"Saved games and settings: {output}")


def main():
    torch.manual_seed(SEED)
    network, cfg, device = load_model()
    player = Player(network, cfg, device)
    if MODE == "play":
        serve(player)
    elif MODE == "training":
        show_training(network, cfg)
    elif MODE == "benchmark":
        benchmark(player)
    else:
        raise ValueError('MODE must be "play", "training", or "benchmark".')


if __name__ == "__main__":
    main()
