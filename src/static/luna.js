"use strict";

const API = "/api/v1";
const FILES = ["a", "b", "c", "d", "e", "f", "g", "h"];
const PIECES = {
    P: "♙", N: "♘", B: "♗", R: "♖", Q: "♕", K: "♔",
    p: "♟", n: "♞", b: "♝", r: "♜", q: "♛", k: "♚",
};
const PIECE_NAMES = { p: "pawn", n: "knight", b: "bishop", r: "rook", q: "queen", k: "king" };
const PROMOTION_ORDER = ["q", "r", "b", "n"];

const ui = {
    game: null,
    health: null,
    orientation: "white",
    selected: null,
    dragSource: null,
    hintMove: null,
    pending: false,
    requestGeneration: 0,
    autoplay: false,
    sound: localStorage.getItem("lunaSound") !== "off",
    audioContext: null,
};

const elements = {};

class ApiRequestError extends Error {
    constructor(status, code, message, details = null) {
        super(message);
        this.status = status;
        this.code = code;
        this.details = details;
    }
}

function collectElements() {
    [
        "lobbyView", "gameView", "missionForm", "strengthSelect", "strengthDescription",
        "launchButton", "spectateButton", "connectionPill", "connectionText", "modelStatus",
        "searchStatus", "brandButton", "newMissionButton", "modeEyebrow", "gameTitle",
        "thinkingOrb", "turnStatus", "statusDetail", "evaluationValue", "evaluationFill",
        "thinkTime", "simulationCount", "confidenceValue", "whiteCaptures", "blackCaptures",
        "topAvatar", "topPlayer", "topPlayerMeta", "topTurnLamp", "bottomAvatar", "bottomPlayer",
        "bottomPlayerMeta", "bottomTurnLamp", "boardShell", "chessboard", "boardThinking",
        "undoButton", "hintButton", "flipButton", "soundButton", "stepButton", "autoplayButton",
        "moveList", "moveCount", "hintCard", "hintSan", "hintUci", "hintMeta", "dismissHint",
        "promotionDialog", "promotionChoices", "toastStack", "liveRegion",
    ].forEach((id) => { elements[id] = document.getElementById(id); });
}

async function api(path, options = {}) {
    const requestOptions = { ...options, headers: { Accept: "application/json", ...(options.headers || {}) } };
    if (requestOptions.body && typeof requestOptions.body !== "string") {
        requestOptions.headers["Content-Type"] = "application/json";
        requestOptions.body = JSON.stringify(requestOptions.body);
    }
    let response;
    try {
        response = await fetch(`${API}${path}`, requestOptions);
    } catch (error) {
        const networkError = new ApiRequestError(0, "network_error", "The observatory could not reach Luna.");
        networkError.cause = error;
        throw networkError;
    }
    const contentType = response.headers.get("content-type") || "";
    const payload = contentType.includes("application/json") ? await response.json() : null;
    if (!response.ok) {
        const failure = payload?.error || {};
        throw new ApiRequestError(
            response.status,
            failure.code || "request_failed",
            failure.message || `Request failed with status ${response.status}.`,
            failure.details || null,
        );
    }
    return payload;
}

function gameRequestContext() {
    return { gameId: ui.game?.id || null, generation: ui.requestGeneration };
}

function isCurrentGameRequest(context) {
    return Boolean(
        context.gameId
        && ui.game?.id === context.gameId
        && ui.requestGeneration === context.generation,
    );
}

function setConnection(state, text) {
    elements.connectionPill.classList.remove("online", "offline");
    if (state) elements.connectionPill.classList.add(state);
    elements.connectionText.textContent = text;
}

async function initialize() {
    collectElements();
    bindEvents();
    elements.soundButton.setAttribute("aria-pressed", String(ui.sound));
    try {
        const response = await api("/health");
        ui.health = response.data;
        setConnection("online", "Model online");
        elements.modelStatus.textContent = "ready";
        populateStrengths(ui.health.strengths);
        elements.launchButton.disabled = false;
        elements.spectateButton.disabled = false;

        const savedGameId = localStorage.getItem("lunaGameId");
        if (savedGameId) {
            try {
                const saved = await api(`/games/${savedGameId}`);
                ui.game = saved.data;
                ui.orientation = localStorage.getItem("lunaOrientation")
                    || ui.game.human_color
                    || "white";
                showGame();
            } catch (error) {
                localStorage.removeItem("lunaGameId");
                if (!(error instanceof ApiRequestError && [404, 410].includes(error.status))) throw error;
            }
        }
    } catch (error) {
        setConnection("offline", "Model offline");
        elements.modelStatus.textContent = "offline";
        elements.searchStatus.textContent = "unavailable";
        showToast(error.message || "Luna is unavailable.");
    }
}

function populateStrengths(strengths) {
    elements.strengthSelect.replaceChildren();
    strengths.forEach((profile) => {
        const option = document.createElement("option");
        option.value = profile.id;
        option.textContent = `${profile.name} · ${profile.simulations} searches`;
        option.dataset.description = profile.description;
        if (profile.id === "strong") option.selected = true;
        elements.strengthSelect.append(option);
    });
    elements.strengthSelect.disabled = false;
    updateStrengthDescription();
}

function updateStrengthDescription() {
    const selected = elements.strengthSelect.selectedOptions[0];
    elements.strengthDescription.textContent = selected?.dataset.description || "Live neural tree search.";
    const selectedProfile = ui.health?.strengths.find((item) => item.id === elements.strengthSelect.value);
    elements.searchStatus.textContent = selectedProfile ? `${selectedProfile.simulations} simulations` : "ready";
}

function bindEvents() {
    elements.missionForm.addEventListener("submit", (event) => {
        event.preventDefault();
        startGame("human");
    });
    elements.spectateButton.addEventListener("click", () => startGame("selfplay"));
    elements.strengthSelect.addEventListener("change", updateStrengthDescription);
    elements.newMissionButton.addEventListener("click", returnToLobby);
    elements.brandButton.addEventListener("click", () => {
        if (ui.game) returnToLobby();
    });
    elements.flipButton.addEventListener("click", flipBoard);
    elements.soundButton.addEventListener("click", toggleSound);
    elements.undoButton.addEventListener("click", undoMove);
    elements.hintButton.addEventListener("click", requestHint);
    elements.stepButton.addEventListener("click", stepSelfplay);
    elements.autoplayButton.addEventListener("click", toggleAutoplay);
    elements.dismissHint.addEventListener("click", clearHint);
    elements.promotionDialog.addEventListener("cancel", () => elements.promotionDialog.close("cancel"));
}

async function startGame(mode) {
    if (ui.pending) return;
    const generation = ++ui.requestGeneration;
    const oldId = ui.game?.id;
    const color = document.querySelector('input[name="color"]:checked')?.value || "white";
    const strength = elements.strengthSelect.value;
    setLaunchPending(true, mode === "selfplay" ? "Opening observatory…" : "Establishing position…");
    try {
        const response = await api("/games", { method: "POST", body: { mode, color, strength } });
        if (generation !== ui.requestGeneration) {
            api(`/games/${response.data.id}`, { method: "DELETE" }).catch(() => {});
            return;
        }
        ui.game = response.data;
        ui.orientation = mode === "human" ? ui.game.human_color : "white";
        ui.selected = null;
        ui.hintMove = null;
        localStorage.setItem("lunaGameId", ui.game.id);
        localStorage.setItem("lunaOrientation", ui.orientation);
        if (oldId && oldId !== ui.game.id) api(`/games/${oldId}`, { method: "DELETE" }).catch(() => {});
        showGame();
        playSound("start");
    } catch (error) {
        if (generation === ui.requestGeneration) handleError(error);
    } finally {
        if (generation === ui.requestGeneration) {
            setLaunchPending(false);
            if (ui.game && !elements.gameView.hidden) render();
        }
    }
}

function setLaunchPending(pending, label = "Launch challenge") {
    ui.pending = pending;
    elements.launchButton.disabled = pending || !ui.health;
    elements.spectateButton.disabled = pending || !ui.health;
    elements.launchButton.querySelector("span").textContent = pending ? label : "Launch challenge";
}

function showGame() {
    if (!ui.game) return;
    elements.lobbyView.hidden = true;
    elements.gameView.hidden = false;
    elements.modeEyebrow.textContent = ui.game.mode === "selfplay" ? "Live neural observatory" : "Live encounter";
    elements.gameTitle.innerHTML = ui.game.mode === "selfplay" ? "Luna <span>observes</span> Luna" : "You <span>vs</span> Luna";
    render();
}

function returnToLobby() {
    ui.requestGeneration += 1;
    stopAutoplay();
    if (elements.promotionDialog.open) elements.promotionDialog.close("cancel");
    setBusy(false);
    const oldId = ui.game?.id;
    ui.game = null;
    ui.selected = null;
    ui.hintMove = null;
    localStorage.removeItem("lunaGameId");
    localStorage.removeItem("lunaOrientation");
    elements.gameView.hidden = true;
    elements.lobbyView.hidden = false;
    clearHint();
    if (oldId) api(`/games/${oldId}`, { method: "DELETE" }).catch(() => {});
}

function parseFen(fen) {
    const pieces = new Map();
    const rows = fen.split(" ")[0].split("/");
    rows.forEach((row, rowIndex) => {
        let file = 0;
        for (const character of row) {
            if (/\d/.test(character)) {
                file += Number(character);
            } else {
                const rank = 8 - rowIndex;
                pieces.set(`${FILES[file]}${rank}`, character);
                file += 1;
            }
        }
    });
    return pieces;
}

function displayedSquares() {
    const files = ui.orientation === "white" ? FILES : [...FILES].reverse();
    const ranks = ui.orientation === "white" ? [8, 7, 6, 5, 4, 3, 2, 1] : [1, 2, 3, 4, 5, 6, 7, 8];
    return ranks.flatMap((rank) => files.map((file) => `${file}${rank}`));
}

function colorOfSymbol(symbol) {
    return symbol === symbol.toUpperCase() ? "white" : "black";
}

function legalMovesFrom(square) {
    return ui.game?.legal_moves.filter((move) => move.slice(0, 2) === square) || [];
}

function isInteractive() {
    return Boolean(
        ui.game
        && ui.game.mode === "human"
        && !ui.pending
        && !ui.game.is_game_over
        && ui.game.turn === ui.game.human_color,
    );
}

function renderBoard() {
    const pieces = parseFen(ui.game.fen);
    const lastSquares = ui.game.last_move ? [ui.game.last_move.slice(0, 2), ui.game.last_move.slice(2, 4)] : [];
    const hintSquares = ui.hintMove ? [ui.hintMove.slice(0, 2), ui.hintMove.slice(2, 4)] : [];
    const selectedTargets = new Set(legalMovesFrom(ui.selected).map((move) => move.slice(2, 4)));
    const currentKing = ui.game.is_check
        ? [...pieces.entries()].find(([, symbol]) => symbol.toLowerCase() === "k" && colorOfSymbol(symbol) === ui.game.turn)?.[0]
        : null;
    const squares = displayedSquares();
    const fragment = document.createDocumentFragment();

    squares.forEach((square, displayIndex) => {
        const fileIndex = FILES.indexOf(square[0]);
        const rank = Number(square[1]);
        const symbol = pieces.get(square);
        const squareButton = document.createElement("button");
        squareButton.type = "button";
        squareButton.dataset.square = square;
        squareButton.className = `square ${(fileIndex + rank) % 2 === 0 ? "light" : "dark"}`;
        squareButton.setAttribute("role", "gridcell");

        if (lastSquares.includes(square)) squareButton.classList.add("last");
        if (ui.selected === square) squareButton.classList.add("selected");
        if (selectedTargets.has(square)) squareButton.classList.add("legal-target");
        if (currentKing === square) squareButton.classList.add("in-check");
        if (hintSquares[0] === square) squareButton.classList.add("hint-source");
        if (hintSquares[1] === square) squareButton.classList.add("hint-target");
        if (symbol) squareButton.classList.add("has-piece");

        const movable = Boolean(symbol && isInteractive() && legalMovesFrom(square).length);
        if (movable) {
            squareButton.classList.add("movable");
            squareButton.draggable = true;
        }
        const description = symbol
            ? `${square}, ${colorOfSymbol(symbol)} ${PIECE_NAMES[symbol.toLowerCase()]}`
            : `${square}, empty`;
        squareButton.setAttribute("aria-label", `${description}${movable ? ", movable" : ""}`);

        if (symbol) {
            const piece = document.createElement("span");
            piece.className = `piece ${colorOfSymbol(symbol)}-piece`;
            piece.textContent = PIECES[symbol];
            piece.setAttribute("aria-hidden", "true");
            squareButton.append(piece);
        }
        if (displayIndex % 8 === 0) {
            const rankLabel = document.createElement("span");
            rankLabel.className = "coordinate rank";
            rankLabel.textContent = square[1];
            squareButton.append(rankLabel);
        }
        if (displayIndex >= 56) {
            const fileLabel = document.createElement("span");
            fileLabel.className = "coordinate file";
            fileLabel.textContent = square[0];
            squareButton.append(fileLabel);
        }

        squareButton.addEventListener("click", handleSquareClick);
        squareButton.addEventListener("dragstart", handleDragStart);
        squareButton.addEventListener("dragover", handleDragOver);
        squareButton.addEventListener("drop", handleDrop);
        squareButton.addEventListener("dragend", () => { ui.dragSource = null; });
        fragment.append(squareButton);
    });
    elements.chessboard.replaceChildren(fragment);
}

function handleSquareClick(event) {
    if (!isInteractive()) return;
    const square = event.currentTarget.dataset.square;
    if (ui.selected) {
        const possible = legalMovesFrom(ui.selected).some((move) => move.slice(2, 4) === square);
        if (possible) {
            submitHumanMove(ui.selected, square);
            return;
        }
        ui.selected = ui.selected === square ? null : (legalMovesFrom(square).length ? square : null);
    } else if (legalMovesFrom(square).length) {
        ui.selected = square;
    }
    renderBoard();
}

function handleDragStart(event) {
    const source = event.currentTarget.dataset.square;
    if (!isInteractive() || !legalMovesFrom(source).length) {
        event.preventDefault();
        return;
    }
    ui.dragSource = source;
    event.dataTransfer.effectAllowed = "move";
    event.dataTransfer.setData("text/plain", source);
    ui.selected = source;
    requestAnimationFrame(renderBoard);
}

function handleDragOver(event) {
    const source = ui.dragSource || event.dataTransfer.getData("text/plain");
    const target = event.currentTarget.dataset.square;
    if (legalMovesFrom(source).some((move) => move.slice(2, 4) === target)) {
        event.preventDefault();
        event.dataTransfer.dropEffect = "move";
    }
}

function handleDrop(event) {
    event.preventDefault();
    const source = ui.dragSource || event.dataTransfer.getData("text/plain");
    const target = event.currentTarget.dataset.square;
    ui.dragSource = null;
    if (legalMovesFrom(source).some((move) => move.slice(2, 4) === target)) {
        submitHumanMove(source, target);
    } else {
        ui.selected = null;
        renderBoard();
    }
}

async function choosePromotion(moves) {
    const color = ui.game.turn;
    elements.promotionChoices.replaceChildren();
    PROMOTION_ORDER.forEach((pieceName) => {
        const move = moves.find((candidate) => candidate.endsWith(pieceName));
        if (!move) return;
        const button = document.createElement("button");
        button.type = "button";
        button.value = move;
        const symbol = color === "white" ? pieceName.toUpperCase() : pieceName;
        button.textContent = PIECES[symbol];
        button.setAttribute("aria-label", `Promote to ${PIECE_NAMES[pieceName]}`);
        button.addEventListener("click", () => elements.promotionDialog.close(move));
        elements.promotionChoices.append(button);
    });
    elements.promotionDialog.showModal();
    return new Promise((resolve) => {
        elements.promotionDialog.addEventListener("close", () => resolve(elements.promotionDialog.returnValue), { once: true });
    });
}

async function submitHumanMove(source, target) {
    if (ui.pending || !ui.game) return;
    const context = gameRequestContext();
    const candidates = legalMovesFrom(source).filter((move) => move.slice(2, 4) === target);
    if (!candidates.length) return;
    let move = candidates[0];
    if (candidates.some((candidate) => candidate.length === 5)) {
        move = await choosePromotion(candidates);
        if (!isCurrentGameRequest(context) || !move || move === "cancel") {
            ui.selected = null;
            if (ui.game) renderBoard();
            return;
        }
    }

    ui.selected = null;
    clearHint();
    setBusy(true, "Luna is calculating");
    try {
        const response = await api(`/games/${context.gameId}/moves`, { method: "POST", body: { move } });
        if (!isCurrentGameRequest(context)) return;
        ui.game = response.data;
        render();
        playEvents(response.events);
        const phrase = response.events.map((event) => `${event.actor} played ${event.san}`).join(". ");
        announce(phrase);
    } catch (error) {
        if (isCurrentGameRequest(context)) {
            handleError(error);
            renderBoard();
        }
    } finally {
        if (isCurrentGameRequest(context)) {
            setBusy(false);
            renderStatus();
            updateControls();
        }
    }
}

function render() {
    if (!ui.game) return;
    renderBoard();
    renderStatus();
    renderPlayers();
    renderEvaluation();
    renderCaptures();
    renderHistory();
    updateControls();
}

function renderStatus() {
    if (!ui.game) return;
    if (ui.pending) {
        elements.turnStatus.textContent = "Calculating";
        elements.statusDetail.textContent = "Searching latent continuations";
        return;
    }
    elements.turnStatus.textContent = ui.game.status;
    if (ui.game.result) {
        elements.statusDetail.textContent = `${ui.game.result.notation} · ${capitalize(ui.game.result.reason)}`;
    } else {
        elements.statusDetail.textContent = `${capitalize(ui.game.turn)} to move${ui.game.is_check ? " · Check" : ""}`;
    }
}

function participant(color) {
    if (ui.game.mode === "selfplay") {
        return { name: `Luna · ${capitalize(color)}`, meta: "Neural engine", avatar: color === "white" ? "Lᴡ" : "Lʙ", human: false };
    }
    if (color === ui.game.human_color) {
        return { name: "You", meta: `${capitalize(color)} pieces`, avatar: "Y", human: true };
    }
    return { name: "Luna", meta: `${capitalize(color)} · Neural engine`, avatar: "L", human: false };
}

function renderPlayers() {
    const bottomColor = ui.orientation;
    const topColor = bottomColor === "white" ? "black" : "white";
    const top = participant(topColor);
    const bottom = participant(bottomColor);
    elements.topAvatar.textContent = top.avatar;
    elements.topAvatar.classList.toggle("human", top.human);
    elements.topPlayer.textContent = top.name;
    elements.topPlayerMeta.textContent = top.meta;
    elements.bottomAvatar.textContent = bottom.avatar;
    elements.bottomAvatar.classList.toggle("human", bottom.human);
    elements.bottomPlayer.textContent = bottom.name;
    elements.bottomPlayerMeta.textContent = bottom.meta;
    elements.topTurnLamp.classList.toggle("active", !ui.game.is_game_over && ui.game.turn === topColor);
    elements.bottomTurnLamp.classList.toggle("active", !ui.game.is_game_over && ui.game.turn === bottomColor);
    elements.topTurnLamp.setAttribute("aria-label", ui.game.turn === topColor ? "Active turn" : "Not active");
    elements.bottomTurnLamp.setAttribute("aria-label", ui.game.turn === bottomColor ? "Active turn" : "Not active");
}

function renderEvaluation(override = null) {
    const evaluation = override ?? ui.game.engine.evaluation_white;
    const meter = elements.evaluationFill.parentElement;
    if (evaluation === null || evaluation === undefined) {
        elements.evaluationValue.textContent = "—";
        elements.evaluationFill.style.width = "50%";
        meter.setAttribute("aria-valuenow", "0");
    } else {
        const bounded = Math.max(-1, Math.min(1, evaluation));
        elements.evaluationValue.textContent = `${bounded >= 0 ? "+" : ""}${bounded.toFixed(2)}`;
        elements.evaluationFill.style.width = `${(bounded + 1) * 50}%`;
        meter.setAttribute("aria-valuenow", String(bounded));
    }
    const engine = ui.game.engine;
    elements.thinkTime.textContent = engine.think_time_ms == null ? "—" : formatDuration(engine.think_time_ms);
    elements.simulationCount.textContent = `${ui.game.strength.simulations} sims`;
    elements.confidenceValue.textContent = engine.confidence == null ? "—" : `${Math.round(engine.confidence * 100)}%`;
}

function renderCaptures() {
    elements.whiteCaptures.textContent = captureGlyphs(ui.game.captured.white);
    elements.blackCaptures.textContent = captureGlyphs(ui.game.captured.black);
}

function captureGlyphs(symbols) {
    return symbols.length ? symbols.map((symbol) => PIECES[symbol]).join(" ") : "None";
}

function renderHistory() {
    const history = ui.game.history;
    const moveLabel = history.length === 1 ? "1 ply" : `${history.length} plies`;
    elements.moveCount.textContent = moveLabel;
    if (!history.length) {
        elements.moveList.innerHTML = '<div class="empty-log"><span aria-hidden="true">☾</span><p>Your moves will appear here.</p></div>';
        return;
    }
    const fragment = document.createDocumentFragment();
    for (let index = 0; index < history.length; index += 2) {
        const white = history[index];
        const black = history[index + 1];
        const row = document.createElement("div");
        row.className = "move-row";
        const number = document.createElement("span");
        number.className = "move-number";
        number.textContent = `${white.move_number}.`;
        row.append(number, moveCell(white, history.length), moveCell(black, history.length));
        fragment.append(row);
    }
    elements.moveList.replaceChildren(fragment);
    requestAnimationFrame(() => { elements.moveList.scrollTop = elements.moveList.scrollHeight; });
}

function moveCell(entry, total) {
    const cell = document.createElement("span");
    cell.className = "move-san";
    if (entry) {
        cell.textContent = entry.san;
        cell.title = entry.uci;
        if (entry.ply === total) cell.classList.add("latest");
    }
    return cell;
}

function updateControls() {
    if (!ui.game) return;
    const isSelfplay = ui.game.mode === "selfplay";
    elements.undoButton.hidden = isSelfplay;
    elements.hintButton.hidden = isSelfplay;
    elements.undoButton.disabled = ui.pending || !ui.game.can_undo;
    elements.hintButton.disabled = ui.pending || !ui.game.can_hint;
    elements.stepButton.hidden = !isSelfplay;
    elements.autoplayButton.hidden = !isSelfplay;
    elements.stepButton.disabled = ui.pending || ui.game.is_game_over || ui.autoplay;
    elements.autoplayButton.disabled = ui.pending && !ui.autoplay;
    elements.autoplayButton.setAttribute("aria-pressed", String(ui.autoplay));
    elements.autoplayButton.setAttribute("aria-label", ui.autoplay ? "Pause automatic self-play" : "Start automatic self-play");
    elements.autoplayButton.querySelector("span:first-child").textContent = ui.autoplay ? "Ⅱ" : "▶";
    elements.autoplayButton.querySelector("span:last-child").textContent = ui.autoplay ? "Pause" : "Auto";
}

function setBusy(busy, message = "Luna is calculating") {
    ui.pending = busy;
    elements.boardThinking.hidden = !busy;
    elements.boardThinking.querySelector("strong").textContent = message;
    document.body.classList.toggle("thinking", busy);
    elements.chessboard.setAttribute("aria-busy", String(busy));
    if (ui.game) {
        updateControls();
        if (!busy) renderBoard();
    }
}

async function undoMove() {
    if (ui.pending || !ui.game?.can_undo) return;
    const context = gameRequestContext();
    stopAutoplay();
    clearHint();
    setBusy(true, "Rewinding the position");
    try {
        const response = await api(`/games/${context.gameId}/undo`, { method: "POST", body: {} });
        if (!isCurrentGameRequest(context)) return;
        ui.game = response.data;
        render();
        playSound("undo");
        announce("Previous turn undone.");
    } catch (error) {
        if (isCurrentGameRequest(context)) handleError(error);
    } finally {
        if (isCurrentGameRequest(context)) {
            setBusy(false);
            renderStatus();
            updateControls();
        }
    }
}

async function requestHint() {
    if (ui.pending || !ui.game?.can_hint) return;
    const context = gameRequestContext();
    clearHint();
    setBusy(true, "Calculating a hint");
    try {
        const response = await api(`/games/${context.gameId}/hint`, { method: "POST", body: {} });
        if (!isCurrentGameRequest(context)) return;
        const hint = response.data;
        ui.hintMove = hint.move;
        ui.game.engine.evaluation_white = hint.evaluation_white;
        ui.game.engine.think_time_ms = hint.think_time_ms;
        ui.game.engine.confidence = hint.confidence;
        elements.hintSan.textContent = hint.san;
        elements.hintUci.textContent = hint.move;
        elements.hintMeta.textContent = `${Math.round(hint.confidence * 100)}% improved search policy · ${formatDuration(hint.think_time_ms)}`;
        elements.hintCard.hidden = false;
        renderBoard();
        renderEvaluation(hint.evaluation_white);
        playSound("hint");
        announce(`Luna recommends ${hint.san}.`);
    } catch (error) {
        if (isCurrentGameRequest(context)) handleError(error);
    } finally {
        if (isCurrentGameRequest(context)) {
            setBusy(false);
            renderStatus();
            updateControls();
        }
    }
}

function clearHint() {
    ui.hintMove = null;
    if (elements.hintCard) elements.hintCard.hidden = true;
    if (ui.game && elements.chessboard) renderBoard();
}

async function stepSelfplay() {
    if (ui.pending || !ui.game || ui.game.mode !== "selfplay" || ui.game.is_game_over) return false;
    const context = gameRequestContext();
    clearHint();
    setBusy(true, `${capitalize(ui.game.turn)} is calculating`);
    try {
        const response = await api(`/games/${context.gameId}/engine-move`, { method: "POST", body: {} });
        if (!isCurrentGameRequest(context)) return false;
        ui.game = response.data;
        render();
        playEvents(response.events);
        announce(`${capitalize(response.events[0].san)} played.`);
        if (ui.game.is_game_over) stopAutoplay();
        return true;
    } catch (error) {
        if (isCurrentGameRequest(context)) {
            handleError(error);
            stopAutoplay();
        }
        return false;
    } finally {
        if (isCurrentGameRequest(context)) {
            setBusy(false);
            renderStatus();
            updateControls();
        }
    }
}

function toggleAutoplay() {
    if (ui.autoplay) {
        stopAutoplay();
        return;
    }
    if (!ui.game || ui.game.is_game_over) return;
    ui.autoplay = true;
    updateControls();
    runAutoplay();
}

async function runAutoplay() {
    while (ui.autoplay && ui.game && !ui.game.is_game_over) {
        const moved = await stepSelfplay();
        if (!moved || !ui.autoplay) break;
        await delay(480);
    }
    stopAutoplay();
}

function stopAutoplay() {
    ui.autoplay = false;
    if (ui.game && elements.autoplayButton) updateControls();
}

function flipBoard() {
    if (!ui.game) return;
    ui.orientation = ui.orientation === "white" ? "black" : "white";
    localStorage.setItem("lunaOrientation", ui.orientation);
    ui.selected = null;
    renderBoard();
    renderPlayers();
}

function toggleSound() {
    ui.sound = !ui.sound;
    localStorage.setItem("lunaSound", ui.sound ? "on" : "off");
    elements.soundButton.setAttribute("aria-pressed", String(ui.sound));
    if (ui.sound) playSound("hint");
}

function playEvents(events = []) {
    const last = events[events.length - 1];
    if (!last) return;
    if (ui.game?.is_game_over) playSound("finish");
    else if (last.san.includes("x")) playSound("capture");
    else if (last.san.includes("+")) playSound("check");
    else playSound("move");
}

function playSound(kind) {
    if (!ui.sound) return;
    try {
        ui.audioContext ||= new (window.AudioContext || window.webkitAudioContext)();
        const context = ui.audioContext;
        if (context.state === "suspended") context.resume();
        const now = context.currentTime;
        const patterns = {
            start: [[330, 0], [440, 0.08], [660, 0.16]],
            move: [[260, 0], [330, 0.055]],
            capture: [[190, 0], [145, 0.07]],
            check: [[440, 0], [620, 0.07]],
            finish: [[330, 0], [494, 0.1], [659, 0.2]],
            hint: [[523, 0], [784, 0.08]],
            undo: [[330, 0], [247, 0.07]],
        };
        (patterns[kind] || patterns.move).forEach(([frequency, offset]) => {
            const oscillator = context.createOscillator();
            const gain = context.createGain();
            oscillator.type = "sine";
            oscillator.frequency.value = frequency;
            gain.gain.setValueAtTime(0.0001, now + offset);
            gain.gain.exponentialRampToValueAtTime(0.055, now + offset + 0.008);
            gain.gain.exponentialRampToValueAtTime(0.0001, now + offset + 0.12);
            oscillator.connect(gain).connect(context.destination);
            oscillator.start(now + offset);
            oscillator.stop(now + offset + 0.13);
        });
    } catch (_) {
        ui.sound = false;
    }
}

function handleError(error) {
    const message = error instanceof ApiRequestError ? error.message : "An unexpected observatory error occurred.";
    showToast(message);
    if (error instanceof ApiRequestError && error.status === 0) setConnection("offline", "Connection lost");
}

function showToast(message, type = "error") {
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.textContent = message;
    elements.toastStack.append(toast);
    window.setTimeout(() => toast.remove(), 4600);
}

function announce(message) {
    elements.liveRegion.textContent = "";
    window.setTimeout(() => { elements.liveRegion.textContent = message; }, 20);
}

function formatDuration(milliseconds) {
    if (milliseconds < 1000) return `${milliseconds} ms`;
    return `${(milliseconds / 1000).toFixed(milliseconds < 10000 ? 1 : 0)} s`;
}

function capitalize(value) {
    return value ? value[0].toUpperCase() + value.slice(1) : "";
}

function delay(milliseconds) {
    return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

document.addEventListener("DOMContentLoaded", initialize);
