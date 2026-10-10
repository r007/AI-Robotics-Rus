// Browser-side provider for the RDK X5 HTTP bridge. This file is the focused
// integration module used by the Microduck Sandbox adaptation in the tutorial.
const SLOT_TO_RDK = Object.freeze({
  walk: "walking",
  stand: "stand",
  sitstand: "sitstand",
  roll: "roll",
  kickL: "kick_left",
  kickR: "kick_right",
  groundpick: "groundpick",
  drive: "drive",
  crouch: "crouch",
});

export const RDK_REQUIRED_POLICIES = Object.freeze(Object.values(SLOT_TO_RDK));

export class RdkPolicyProvider {
  static async connect(baseUrl) {
    const response = await fetch(`${baseUrl}/catalog`, { cache: "no-store" });
    if (!response.ok) throw new Error(`RDK catalog HTTP ${response.status}`);
    const catalog = await response.json();
    if (catalog.protocol !== "MDP2") throw new Error(`unsupported RDK protocol: ${catalog.protocol}`);
    const missing = RDK_REQUIRED_POLICIES.filter((name) => !catalog.policies?.[name]);
    if (missing.length) throw new Error(`RDK missing policies: ${missing.join(", ")}`);
    for (const name of RDK_REQUIRED_POLICIES) {
      const contract = catalog.policies[name];
      if (contract.observations !== 61 || contract.actions !== 14) {
        throw new Error(`RDK ${name} contract must be [1,61] -> [1,14]`);
      }
    }
    return new RdkPolicyProvider(baseUrl, catalog);
  }

  constructor(baseUrl, catalog) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.catalog = catalog;
  }

  async run(slot, observation) {
    const policy = SLOT_TO_RDK[slot];
    if (!policy) throw new Error(`no RDK policy mapping for slot: ${slot}`);
    const body = observation instanceof Float32Array
      ? observation.slice().buffer
      : Float32Array.from(observation).buffer;
    const started = performance.now();
    const response = await fetch(`${this.baseUrl}/infer/${encodeURIComponent(policy)}`, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body,
      cache: "no-store",
    });
    if (!response.ok) throw new Error(`RDK ${policy} inference HTTP ${response.status}`);
    const payload = await response.arrayBuffer();
    if (payload.byteLength !== 56) throw new Error(`RDK returned ${payload.byteLength} bytes`);
    const actions = new Float32Array(payload.slice(0));
    if (!actions.every(Number.isFinite)) throw new Error(`RDK ${policy} returned NaN or Inf`);
    return { policy, actions, latencyMs: performance.now() - started };
  }
}
