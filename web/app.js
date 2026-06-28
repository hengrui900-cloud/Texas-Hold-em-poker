const stageNames = ["翻牌前", "翻牌", "转牌", "河牌", "摊牌"];
const actionLabels = {
  fold: "弃牌",
  check_call: "过牌 / 跟注",
  raise_half_pot: "加注半池",
  raise_pot: "加注底池",
  all_in: "全押",
};
const suitText = { S: "♠", H: "♥", D: "♦", C: "♣" };

let currentState = null;
let selectedBet = 0;
let turnTimerTimeout = null;
let turnTimerInterval = null;
let timerEndAt = 0;
let timerTotalMs = 0;
let timerPlayer = null;

async function api(path, payload = null) {
  const options = payload === null
    ? {}
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) };
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok || data.error) {
    throw new Error(data.error || `请求失败：${response.status}`);
  }
  return data;
}

async function loadState() {
  clearTurnTimer();
  currentState = await api("/api/state");
  render();
}

async function newHand() {
  clearTurnTimer();
  selectedBet = 0;
  currentState = await api("/api/new-hand", {});
  render();
}

async function resetGame() {
  clearTurnTimer();
  selectedBet = 0;
  currentState = await api("/api/reset-game", {});
  render();
}

async function act(action) {
  if (!currentState || currentState.terminal || currentState.bankrupt) return;
  if (currentState.current_player !== 0) return;
  clearTurnTimer();
  selectedBet = 0;
  currentState = await api("/api/action", { action });
  render();
}

async function aiActAfterThinking() {
  if (!currentState || currentState.terminal || currentState.bankrupt || currentState.current_player === 0) return;
  clearTurnTimer();
  currentState = await api("/api/ai-action", {});
  render();
}

function render() {
  if (!currentState) return;
  clearTurnTimer();
  document.getElementById("devicePill").textContent = currentState.checkpoint_loaded
    ? "AI：已加载训练模型"
    : "AI：规则兜底";
  document.getElementById("ruleTip").textContent = currentState.rule_tip;
  document.getElementById("ruleTipSide").textContent = currentState.rule_tip;
  document.getElementById("stageLabel").textContent = stageNames[currentState.street_index] || currentState.stage;
  renderStreets();
  renderTable();
  renderBettingDock();
  renderSidebar();
  updateTopActions();
  scheduleTurnTimer();
}

function renderStreets() {
  const html = stageNames
    .slice(0, 4)
    .map((name, index) => {
      const active = currentState.street_index === index;
      return `<div class="street-step ${active ? "active" : ""}"><span class="dot"></span>${name}</div>`;
    })
    .join("");
  document.getElementById("streetSteps").innerHTML = html;
}

function renderTable() {
  document.getElementById("potPill").textContent = `底池 ${money(currentState.pot)}`;
  document.getElementById("communityCards").innerHTML = renderCommunityCards(currentState.public_cards);
  currentState.players.forEach((player) => {
    const node = document.getElementById(`seat${player.id}`);
    if (node) node.innerHTML = renderSeat(player);
  });
}

function renderCommunityCards(cards) {
  const filled = cards.map(renderCard);
  while (filled.length < 5) filled.push('<div class="card-slot"></div>');
  return filled.join("");
}

function renderSeat(player) {
  const active = currentState.current_player === player.id && !currentState.terminal && !currentState.bankrupt;
  const status = player.folded ? "已弃牌" : player.all_in ? "已全押" : active ? "行动中" : "等待";
  const cards = player.cards.length
    ? player.cards.map(renderCard).join("")
    : Array.from({ length: player.hidden_cards }, () => '<div class="card-back"></div>').join("");
  const committed = player.committed ? `<div class="table-bet">已下注 ${money(player.committed)}</div>` : "";
  const timer = active
    ? `<div class="turn-meter" aria-label="${player.name} 行动倒计时">
        <div class="turn-meter-head">
          <span>${player.is_human ? "你的行动时间" : "AI 思考中"}</span>
          <strong id="turnText-${player.id}"></strong>
        </div>
        <div class="turn-track"><div class="turn-fill" id="turnProgress-${player.id}"></div></div>
      </div>`
    : "";
  return `
    <div class="seat-card ${active ? "active" : ""} ${currentState.bankrupt ? "bankrupt" : ""}">
      ${committed}
      <div class="seat-main">
        <div class="avatar ${player.is_human ? "human" : ""}">${player.is_human ? "我" : "AI"}</div>
        <div>
          <div class="name">${player.name}</div>
          <div class="bankroll">${money(player.stack)} 储备</div>
          <div class="status">${player.personality} · ${status}</div>
        </div>
      </div>
      ${timer}
      <div class="chip-reserve" aria-label="${player.name} 筹码储备">${renderChipReserve(player.stack)}</div>
      <div class="seat-cards">${cards}</div>
      ${player.is_dealer ? '<div class="dealer">D</div>' : ""}
    </div>
  `;
}

function renderChipReserve(amount) {
  const rack = currentState.chip_rack || [];
  const totalRackValue = rack.reduce((sum, chip) => sum + chip.value * chip.count, 0);
  if (Math.round(Number(amount || 0)) === totalRackValue) {
    return rack
      .filter((chip) => chip.count > 0)
      .map((chip) => `
        <div class="chip-pile">
          <span class="mini-chip chip-${chip.value}">${money(chip.value)}</span>
          <strong>×${chip.count}</strong>
        </div>
      `)
      .join("");
  }

  const denominations = [...rack].sort((a, b) => b.value - a.value);
  let remaining = Math.max(0, Math.round(Number(amount || 0)));
  const piles = [];
  denominations.forEach((chip) => {
    const count = Math.floor(remaining / chip.value);
    if (count > 0) {
      piles.push(`
        <div class="chip-pile">
          <span class="mini-chip chip-${chip.value}">${money(chip.value)}</span>
          <strong>×${count}</strong>
        </div>
      `);
      remaining -= count * chip.value;
    }
  });
  if (remaining > 0) {
    piles.push(`
      <div class="chip-pile">
        <span class="mini-chip chip-cash">${money(remaining)}</span>
        <strong>零钱</strong>
      </div>
    `);
  }
  return piles.join("") || '<span class="empty-stack">无筹码</span>';
}

function renderCard(card) {
  return `
    <div class="card-face ${card.color}">
      <span>${card.rank}</span>
      <span>${suitText[card.suit] || card.suit}</span>
    </div>
  `;
}

function renderBettingDock() {
  const humanTurn = currentState.current_player === 0 && !currentState.terminal && !currentState.bankrupt;
  const legal = new Set(currentState.legal_actions);
  const player = currentState.players.find((item) => item.id === 0);
  const maxStack = player?.stack || 0;
  selectedBet = Math.min(selectedBet, maxStack);

  document.getElementById("selectedBetText").textContent = money(selectedBet);
  document.getElementById("chipButtons").innerHTML = (currentState.chip_rack || [])
    .map((chip) => {
      const disabled = !humanTurn || selectedBet + chip.value > maxStack;
      return `
        <button class="bet-chip chip-${chip.value}" data-chip="${chip.value}" ${disabled ? "disabled" : ""}>
          <span>${money(chip.value)}</span>
          <small>×${chip.count}</small>
        </button>
      `;
    })
    .join("");

  const callAmount = currentState.action_options.check_call?.amount || 0;
  const mapped = mapSelectedBetToAction();
  document.getElementById("betNote").textContent = selectedBet
    ? `已选择 ${money(selectedBet)}，将执行：${actionLabels[mapped]}。点击几下筹码就下几个注。`
    : callAmount
      ? `请先跟注 ${money(callAmount)}，或点击筹码后加注。`
      : "点击筹码，点击几下就下几个注；不选筹码时可过牌。";

  setActionButton("foldButton", humanTurn && legal.has("fold"));
  setActionButton("callButton", humanTurn && legal.has("check_call"), callAmount ? `跟注 ${money(callAmount)}` : "过牌");
  setActionButton("betButton", humanTurn && Boolean(mapped), selectedBet ? `下注 ${money(selectedBet)}` : "下注所选");
  setActionButton("allInButton", humanTurn && legal.has("all_in"), `全押 ${money(currentState.action_options.all_in?.amount || 0)}`);
  setActionButton("clearBetButton", selectedBet > 0, "清空下注");

  document.querySelectorAll("[data-chip]").forEach((button) => {
    button.addEventListener("click", () => {
      selectedBet = Math.min(maxStack, selectedBet + Number(button.dataset.chip));
      renderBettingDock();
      updateTurnProgress();
    });
  });
}

function setActionButton(id, enabled, label = null) {
  const button = document.getElementById(id);
  button.disabled = !enabled;
  if (label) button.textContent = label;
}

function mapSelectedBetToAction() {
  const legal = new Set(currentState.legal_actions);
  const options = currentState.action_options;
  const callAmount = options.check_call?.amount || 0;
  if (selectedBet > 0 && legal.has("all_in") && selectedBet >= (options.all_in?.amount || Infinity)) return "all_in";
  if (selectedBet > 0 && legal.has("raise_pot") && selectedBet >= (options.raise_pot?.amount || Infinity)) return "raise_pot";
  if (selectedBet > callAmount && legal.has("raise_half_pot")) return "raise_half_pot";
  if (legal.has("check_call")) return "check_call";
  if (legal.has("fold")) return "fold";
  return null;
}

function renderSidebar() {
  const stakes = currentState.stakes;
  document.getElementById("blindLabel").textContent = `小盲 ${money(stakes.small_blind)} / 大盲 ${money(stakes.big_blind)}`;
  document.getElementById("chipLegend").innerHTML = `
    <div class="legend-row"><span>初始筹码</span><strong>${money(stakes.starting_stack)}</strong></div>
    ${(currentState.chip_rack || [])
      .map((chip) => `<div class="legend-row"><span>${money(chip.value)} 筹码</span><strong>×${chip.count}</strong></div>`)
      .join("")}
  `;

  document.getElementById("actionLog").innerHTML =
    currentState.last_actions
      .slice(-7)
      .reverse()
      .map((row) => `<div class="log-row"><span>${row.name} · ${translateStage(row.stage)}</span><strong>${row.label}</strong></div>`)
      .join("") || '<div class="status-line">暂无行动记录</div>';

  renderTrend();
}

function renderTrend() {
  const trend = currentState.win_loss_trend;
  document.getElementById("trendStats").textContent = `${trend.hands} 局 / EV ${Number(trend.ev_per_hand || 0).toFixed(3)}`;
  const values = trend.values.length ? trend.values : [0];
  const min = Math.min(...values, -1);
  const max = Math.max(...values, 1);
  const points = values
    .map((value, index) => {
      const x = values.length === 1 ? 0 : (index / (values.length - 1)) * 300;
      const y = 80 - ((value - min) / (max - min || 1)) * 70;
      return `${x},${y}`;
    })
    .join(" ");
  document.getElementById("trendChart").innerHTML = `
    <svg viewBox="0 0 300 92" role="img" aria-label="胜负趋势">
      <line x1="0" y1="80" x2="300" y2="80" stroke="rgba(255,255,255,.18)" />
      <polyline points="${points}" fill="none" stroke="#63dd75" stroke-width="3" />
    </svg>
  `;
}

function updateTopActions() {
  document.getElementById("newHandTop").disabled = Boolean(currentState.bankrupt);
  document.getElementById("resetGameTop").disabled = false;
}

function scheduleTurnTimer() {
  const timer = currentState.turn_timer;
  if (!timer || currentState.terminal || currentState.bankrupt) {
    updateTurnProgress();
    return;
  }
  timerPlayer = timer.player;
  timerTotalMs = Math.max(1, Number(timer.seconds || 0) * 1000);
  timerEndAt = Date.now() + timerTotalMs;
  updateTurnProgress();
  turnTimerInterval = window.setInterval(updateTurnProgress, 100);
  turnTimerTimeout = window.setTimeout(() => {
    if (timer.player === 0) {
      humanTimerExpired().catch(showError);
    } else {
      aiActAfterThinking().catch(showError);
    }
  }, timerTotalMs);
}

function updateTurnProgress() {
  if (timerPlayer === null || !currentState) return;
  const fill = document.getElementById(`turnProgress-${timerPlayer}`);
  const text = document.getElementById(`turnText-${timerPlayer}`);
  if (!fill && !text) return;
  const remaining = Math.max(0, timerEndAt - Date.now());
  const progress = Math.min(100, Math.max(0, ((timerTotalMs - remaining) / timerTotalMs) * 100));
  if (fill) fill.style.width = `${progress}%`;
  if (text) text.textContent = `${(remaining / 1000).toFixed(1)} 秒`;
}

async function humanTimerExpired() {
  if (!currentState || currentState.terminal || currentState.bankrupt || currentState.current_player !== 0) return;
  const legal = new Set(currentState.legal_actions);
  const callAmount = currentState.action_options.check_call?.amount || 0;
  if (legal.has("check_call") && callAmount === 0) {
    await act("check_call");
    return;
  }
  if (legal.has("fold")) {
    await act("fold");
    return;
  }
  if (legal.has("check_call")) {
    await act("check_call");
  }
}

function clearTurnTimer() {
  if (turnTimerTimeout !== null) window.clearTimeout(turnTimerTimeout);
  if (turnTimerInterval !== null) window.clearInterval(turnTimerInterval);
  turnTimerTimeout = null;
  turnTimerInterval = null;
  timerEndAt = 0;
  timerTotalMs = 0;
  timerPlayer = null;
}

function showError(error) {
  clearTurnTimer();
  const message = error?.stack || String(error);
  document.getElementById("ruleTip").textContent = `出错了：${message}`;
  document.getElementById("ruleTipSide").textContent = `出错了：${message}`;
}

function translateStage(stage) {
  const normalized = String(stage || "").toLowerCase();
  return {
    preflop: "翻牌前",
    flop: "翻牌",
    turn: "转牌",
    river: "河牌",
    showdown: "摊牌",
  }[normalized] || stage;
}

function money(value) {
  return `$${Math.round(Number(value || 0))}`;
}

document.getElementById("newHandTop").addEventListener("click", newHand);
document.getElementById("resetGameTop").addEventListener("click", resetGame);
document.getElementById("reloadState").addEventListener("click", loadState);
document.getElementById("foldButton").addEventListener("click", () => act("fold").catch(showError));
document.getElementById("callButton").addEventListener("click", () => act("check_call").catch(showError));
document.getElementById("betButton").addEventListener("click", () => {
  const mapped = mapSelectedBetToAction();
  if (mapped) act(mapped).catch(showError);
});
document.getElementById("allInButton").addEventListener("click", () => act("all_in").catch(showError));
document.getElementById("clearBetButton").addEventListener("click", () => {
  selectedBet = 0;
  renderBettingDock();
  updateTurnProgress();
});

loadState().catch((error) => {
  document.body.innerHTML = `<pre style="color:white;padding:24px">${error.stack || error}</pre>`;
});
