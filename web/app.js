const stageNames = ["Preflop", "Flop", "Turn", "River", "Showdown"];
const actionOrder = ["fold", "check_call", "raise_half_pot", "raise_pot", "all_in"];
const suitText = { S: "♠", H: "♥", D: "♦", C: "♣" };

let currentState = null;

async function api(path, payload = null) {
  const options = payload
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }
    : {};
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok || data.error) {
    throw new Error(data.error || `Request failed: ${response.status}`);
  }
  return data;
}

async function loadState() {
  currentState = await api("/api/state");
  render();
}

async function newHand() {
  const seed = Number(localStorage.getItem("texas-seed") || 7);
  currentState = await api("/api/new-hand", { ai_count: 3, seed });
  render();
}

async function act(action) {
  if (!currentState || currentState.terminal) return;
  currentState = await api("/api/action", { action });
  render();
}

function render() {
  if (!currentState) return;
  document.getElementById("devicePill").textContent = currentState.checkpoint_loaded
    ? "model: DQN checkpoint"
    : "model: rule fallback";
  renderStreets();
  renderTable();
  renderActions();
  renderInspector();
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
  document.getElementById("streetMini").innerHTML = html;
}

function renderTable() {
  document.getElementById("potPill").textContent = `Pot: ${bb(currentState.pot)}`;
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
  const active = currentState.current_player === player.id && !currentState.terminal;
  const status = player.folded ? "Folded" : player.all_in ? "All-in" : active ? "Thinking" : "Checking";
  const cards = player.cards.length
    ? player.cards.map(renderCard).join("")
    : Array.from({ length: player.hidden_cards }, () => '<div class="card-back"></div>').join("");
  return `
    <div class="seat-card ${active ? "active" : ""}">
      ${player.committed ? `<div class="chip">${bb(player.committed)}</div>` : ""}
      <div class="seat-main">
        <div class="avatar ${player.is_human ? "human" : ""}">${player.is_human ? "YOU" : "AI"}</div>
        <div>
          <div class="name">${player.name}</div>
          <div class="meta">Stack ${bb(player.stack)}</div>
          <div class="status">${player.personality} / ${status}</div>
        </div>
      </div>
      <div class="seat-cards">${cards}</div>
      ${player.is_dealer ? '<div class="dealer">D</div>' : ""}
    </div>
  `;
}

function renderCard(card) {
  return `
    <div class="card-face ${card.color}">
      <span>${card.rank}</span>
      <span>${suitText[card.suit] || card.suit}</span>
    </div>
  `;
}

function renderActions() {
  const legal = new Set(currentState.legal_actions);
  actionOrder.forEach((action) => {
    const button = document.querySelector(`[data-action="${action}"]`);
    const option = currentState.action_options[action];
    if (!button || !option) return;
    button.disabled = currentState.terminal || !legal.has(action);
    const amount = option.amount ? ` <small>${bb(option.amount)}</small>` : "";
    button.innerHTML = `${option.label}${amount}`;
  });
  const potAmount = currentState.action_options.raise_pot?.amount || 0;
  const maxAmount = currentState.action_options.all_in?.amount || 0;
  document.getElementById("raiseAmount").value = bb(potAmount);
  document.getElementById("raiseSlider").max = String(Math.max(1, maxAmount));
  document.getElementById("raiseSlider").value = String(potAmount);
}

function renderInspector() {
  const thinking = currentState.ai_thinking;
  document.getElementById("thinkingActor").textContent = thinking.name || "Waiting";
  document.getElementById("thinkingBody").innerHTML = `
    <div class="kv"><span>Street</span><strong>${thinking.street}</strong></div>
    <div class="kv"><span>Hand Range</span><strong>${thinking.hand_range}</strong></div>
    <div class="kv"><span>Intent</span><strong>${thinking.intent}</strong></div>
    <div class="kv"><span>Confidence</span><strong>${Math.round((thinking.confidence || 0) * 100)}%</strong></div>
  `;

  document.getElementById("legalCount").textContent = `${currentState.legal_actions.length} legal`;
  document.getElementById("legalActions").innerHTML =
    actionOrder
      .map((action) => {
        const option = currentState.action_options[action];
        const enabled = currentState.legal_actions.includes(action);
        return `<div class="legal-row"><span>${option.label}</span><strong>${enabled ? bb(option.amount) : "locked"}</strong></div>`;
      })
      .join("") || '<div class="status-line">No legal actions</div>';

  const qRows = thinking.q_values || [];
  document.getElementById("qValues").innerHTML =
    qRows
      .map((item) => {
        const normalized = Math.max(0, Math.min(100, ((item.value + 1) / 2) * 100));
        return `
          <div class="q-row">
            <span>${item.label}</span>
            <div class="bar-track"><div class="bar-fill" style="width:${normalized}%"></div></div>
            <strong>${Number(item.value).toFixed(2)}</strong>
          </div>
        `;
      })
      .join("") || '<div class="status-line">AI has not acted yet</div>';

  document.getElementById("actionLog").innerHTML =
    currentState.last_actions
      .slice(-5)
      .reverse()
      .map((row) => `<div class="log-row"><span>${row.name} / ${row.stage}</span><strong>${row.label}</strong></div>`)
      .join("") || '<div class="status-line">No actions yet</div>';

  renderTrend();
}

function renderTrend() {
  const trend = currentState.win_loss_trend;
  document.getElementById("trendStats").textContent = `${trend.hands} hands / EV ${trend.ev_per_hand}`;
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
    <svg viewBox="0 0 300 92" role="img" aria-label="win loss trend">
      <line x1="0" y1="80" x2="300" y2="80" stroke="rgba(255,255,255,.18)" />
      <polyline points="${points}" fill="none" stroke="#63dd75" stroke-width="3" />
    </svg>
  `;
}

function bb(value) {
  return `${Number(value || 0).toFixed(1)} BB`;
}

document.querySelectorAll("[data-action]").forEach((button) => {
  button.addEventListener("click", () => act(button.dataset.action));
});
document.getElementById("newHandTop").addEventListener("click", newHand);
document.getElementById("reloadState").addEventListener("click", loadState);
document.getElementById("quickHalf").addEventListener("click", () => act("raise_half_pot"));
document.getElementById("quickPot").addEventListener("click", () => act("raise_pot"));
document.getElementById("quickMax").addEventListener("click", () => act("all_in"));

loadState().catch((error) => {
  document.body.innerHTML = `<pre style="color:white;padding:24px">${error.stack || error}</pre>`;
});
