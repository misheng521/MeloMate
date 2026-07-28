const state = {
  step: 0,
  currentTurn: "user",
  log: [],
};

const stateText = document.querySelector("#stateText");

function render() {
  stateText.textContent = `step=${state.step}, turn=${state.currentTurn}`;
}

function applyUserStep() {
  if (state.currentTurn !== "user") return false;
  state.step += 1;
  state.currentTurn = "melomate";
  state.log.push({ by: "user", step: state.step });
  render();
  return true;
}

function applyMeloMateStep() {
  if (state.currentTurn !== "melomate") return { handled: true, accepted: false };
  state.step += 1;
  state.currentTurn = "user";
  state.log.push({ by: "melomate", step: state.step });
  render();
  return { handled: true, accepted: true, result: { step: state.step } };
}

document.querySelector("#userStep").addEventListener("click", applyUserStep);
document.querySelector("#reset").addEventListener("click", () => {
  state.step = 0;
  state.currentTurn = "user";
  state.log = [];
  render();
});

window.MeloMateGameState = () => ({
  screen: "template",
  step: state.step,
  currentTurn: state.currentTurn,
  log: state.log,
  availableActions: state.currentTurn === "melomate" ? [{ action: "take-step", payload: {} }] : [],
});

window.MeloMateGameAction = (action) => {
  if (action !== "take-step") return { handled: false, accepted: false };
  return applyMeloMateStep();
};

render();
