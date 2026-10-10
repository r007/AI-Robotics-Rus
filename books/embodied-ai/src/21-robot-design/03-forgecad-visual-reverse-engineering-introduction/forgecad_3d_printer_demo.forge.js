// ForgeCAD reconstruction demo inspired by the public 3dprinter-gpt52codex
// benchmark prompt. The official benchmark publishes a GIF artifact; this
// script is a local, runnable reconstruction for tutorial use.

const frameWidth = Param.number("Frame Width", 180, { min: 140, max: 240, unit: "mm" });
const frameDepth = Param.number("Frame Depth", 150, { min: 120, max: 220, unit: "mm" });
const frameHeight = Param.number("Frame Height", 210, { min: 160, max: 280, unit: "mm" });
const headX = Param.number("Print Head X", 22, { min: -70, max: 70, unit: "mm" });
const bedY = Param.number("Print Bed Y", 12, { min: -45, max: 45, unit: "mm" });
const gantryZ = Param.number("Gantry Z", 128, { min: 70, max: 175, unit: "mm" });
const buildVolumeW = Param.number("Build Volume Width", 118, { min: 80, max: 150, unit: "mm" });
const buildVolumeD = Param.number("Build Volume Depth", 105, { min: 80, max: 135, unit: "mm" });

const bar = 8;
const halfW = frameWidth / 2;
const halfD = frameDepth / 2;

const colors = {
  frame: "#0f1012",
  rail: "#cfd3d7",
  bed: "#2c7fbd",
  carriage: "#202327",
  nozzle: "#d58a2a",
  belt: "#050607",
  electronics: "#15171a",
  spool: "#b91f2f",
  filament: "#e8e8e8",
  guide: "#23c9ea",
};

function beamX(length, y, z, color = colors.frame) {
  return box(length, bar, bar).translate(0, y, z).color(color);
}

function beamY(length, x, z, color = colors.frame) {
  return box(bar, length, bar).translate(x, 0, z).color(color);
}

function post(x, y, height, color = colors.frame) {
  return box(bar, bar, height).translate(x, y, 0).color(color);
}

function rodX(length, radius, y, z, color = colors.rail) {
  return cylinder(length, radius, undefined, 32)
    .pointAlong([1, 0, 0])
    .translate(0, y, z)
    .color(color)
    .material({ metalness: 0.82, roughness: 0.18 });
}

function rodY(length, radius, x, z, color = colors.rail) {
  return cylinder(length, radius, undefined, 32)
    .pointAlong([0, 1, 0])
    .translate(x, 0, z)
    .color(color)
    .material({ metalness: 0.82, roughness: 0.18 });
}

function rodZ(height, radius, x, y, color = colors.rail) {
  return cylinder(height, radius, undefined, 32)
    .translate(x, y, 0)
    .color(color)
    .material({ metalness: 0.82, roughness: 0.18 });
}

function foot(x, y) {
  return box(24, 18, 6).translate(x, y, -3).color("#17191d");
}

function pulley(x, y, z) {
  const wheel = cylinder(6, 8, undefined, 48)
    .pointAlong([0, 1, 0])
    .translate(x, y, z)
    .color("#1d2024");
  const cap = cylinder(8, 2.2, undefined, 32)
    .pointAlong([0, 1, 0])
    .translate(x, y, z)
    .color(colors.rail);
  return union(wheel, cap);
}

function buildVolumeCage(w, d, h, centerY, z0) {
  const x0 = -w / 2;
  const x1 = w / 2;
  const y0 = centerY - d / 2;
  const y1 = centerY + d / 2;
  const z1 = z0 + h;
  const r = 1.0;
  const edge = colors.guide;
  return union(
    cylinder(h, r, undefined, 12).translate(x0, y0, z0).color(edge),
    cylinder(h, r, undefined, 12).translate(x1, y0, z0).color(edge),
    cylinder(h, r, undefined, 12).translate(x0, y1, z0).color(edge),
    cylinder(h, r, undefined, 12).translate(x1, y1, z0).color(edge),
    box(w, r * 2, r * 2).translate(0, y0, z0).color(edge),
    box(w, r * 2, r * 2).translate(0, y1, z0).color(edge),
    box(r * 2, d, r * 2).translate(x0, centerY, z0).color(edge),
    box(r * 2, d, r * 2).translate(x1, centerY, z0).color(edge),
    box(w, r * 2, r * 2).translate(0, y0, z1).color(edge),
    box(w, r * 2, r * 2).translate(0, y1, z1).color(edge),
    box(r * 2, d, r * 2).translate(x0, centerY, z1).color(edge),
    box(r * 2, d, r * 2).translate(x1, centerY, z1).color(edge),
  ).material({ emissive: edge, emissiveIntensity: 0.2, opacity: 0.86 });
}

const frameParts = [
  beamX(frameWidth, -halfD, bar / 2),
  beamX(frameWidth, halfD, bar / 2),
  beamY(frameDepth, -halfW, bar / 2),
  beamY(frameDepth, halfW, bar / 2),
  beamX(frameWidth, -halfD, frameHeight),
  beamX(frameWidth, halfD, frameHeight),
  beamY(frameDepth, -halfW, frameHeight),
  beamY(frameDepth, halfW, frameHeight),
  post(-halfW, -halfD, frameHeight),
  post(halfW, -halfD, frameHeight),
  post(-halfW, halfD, frameHeight),
  post(halfW, halfD, frameHeight),
  foot(-halfW, -halfD),
  foot(halfW, -halfD),
  foot(-halfW, halfD),
  foot(halfW, halfD),
];

const bedPlate = box(buildVolumeW, buildVolumeD, 5)
  .translate(0, bedY, 26)
  .color(colors.bed)
  .material({ metalness: 0.15, roughness: 0.36 });
const heatedPlate = box(buildVolumeW + 18, buildVolumeD + 18, 4)
  .translate(0, bedY, 20)
  .color("#b85c20");
const bedCarriage = box(buildVolumeW * 0.75, 20, 8).translate(0, bedY - 4, 14).color("#30343a");
const bedRails = union(
  rodY(frameDepth - 34, 2.2, -buildVolumeW * 0.38, 18),
  rodY(frameDepth - 34, 2.2, buildVolumeW * 0.38, 18),
);

const zMotion = union(
  rodZ(frameHeight - 28, 2.2, -halfW + 26, -halfD + 20),
  rodZ(frameHeight - 28, 2.2, halfW - 26, -halfD + 20),
  rodZ(frameHeight - 42, 2.8, -halfW + 38, halfD - 24),
  rodZ(frameHeight - 42, 2.8, halfW - 38, halfD - 24),
);

const gantry = union(
  beamX(frameWidth - 46, -halfD + 28, gantryZ, "#101114"),
  rodX(frameWidth - 58, 2.4, -halfD + 20, gantryZ + 9),
  rodX(frameWidth - 58, 2.4, -halfD + 38, gantryZ + 9),
  box(18, 18, 22).translate(-halfW + 24, -halfD + 28, gantryZ - 6).color("#181a1f"),
  box(18, 18, 22).translate(halfW - 24, -halfD + 28, gantryZ - 6).color("#181a1f"),
);

const carriageBody = box(28, 24, 28).translate(headX, -halfD + 29, gantryZ + 3).color(colors.carriage);
const fan = cylinder(5, 10, undefined, 48)
  .pointAlong([0, 1, 0])
  .translate(headX, -halfD + 15, gantryZ + 7)
  .color("#0b0c0e");
const hotend = cylinder(16, 4.2, 2.2, 32).translate(headX, -halfD + 29, gantryZ - 22).color(colors.nozzle);
const nozzle = cylinder(9, 3.2, 0.8, 32).translate(headX, -halfD + 29, gantryZ - 33).color("#bf6f1b");
const heatsink = union(
  box(18, 16, 2).translate(headX, -halfD + 29, gantryZ - 7).color("#c7cbd0"),
  box(18, 16, 2).translate(headX, -halfD + 29, gantryZ - 12).color("#c7cbd0"),
  box(18, 16, 2).translate(headX, -halfD + 29, gantryZ - 17).color("#c7cbd0"),
);

const belts = union(
  box(frameWidth - 62, 2.2, 2.2).translate(0, -halfD + 13, gantryZ + 20).color(colors.belt),
  box(frameWidth - 62, 2.2, 2.2).translate(0, -halfD + 45, gantryZ + 20).color(colors.belt),
  pulley(-halfW + 23, -halfD + 29, gantryZ + 20),
  pulley(halfW - 23, -halfD + 29, gantryZ + 20),
);

const spoolX = halfW - 24;
const spoolY = halfD + 13;
const spoolZ = frameHeight - 8;
const spool = union(
  cylinder(22, 20, undefined, 64).pointAlong([1, 0, 0]).translate(spoolX, spoolY, spoolZ).color(colors.spool),
  cylinder(4, 24, undefined, 64).pointAlong([1, 0, 0]).translate(spoolX - 13, spoolY, spoolZ).color("#d8d8d8"),
  cylinder(4, 24, undefined, 64).pointAlong([1, 0, 0]).translate(spoolX + 13, spoolY, spoolZ).color("#d8d8d8"),
  cylinder(30, 5, undefined, 40).pointAlong([1, 0, 0]).translate(spoolX, spoolY, spoolZ).color("#3a3d42"),
  box(62, 5, 5).translate(spoolX - 18, halfD + 5, spoolZ).color("#17191d"),
  box(7, 14, 30).translate(spoolX - 50, halfD + 1, spoolZ - 18).color("#17191d"),
  box(7, 14, 30).translate(spoolX + 10, halfD + 1, spoolZ - 18).color("#17191d"),
);

const filamentPath = union(
  cylinder(42, 1.2, undefined, 16).translate(spoolX - 8, spoolY, spoolZ - 42).color(colors.filament),
  cylinder(frameDepth - 62, 1.2, undefined, 16).pointAlong([0, 1, 0]).translate(headX, 10, gantryZ + 34).color(colors.filament),
  cylinder(38, 1.2, undefined, 16).translate(headX, -halfD + 29, gantryZ - 4).color(colors.filament),
);

const electronics = union(
  box(44, 34, 28).translate(0, -halfD - 14, 18).color(colors.electronics),
  box(26, 2.4, 14).translate(0, -halfD - 31.5, 23).color("#090b0d"),
  cylinder(3, 3.2, undefined, 32).pointAlong([0, 1, 0]).translate(-12, -halfD - 33, 23).color(colors.guide),
  cylinder(3, 3.2, undefined, 32).pointAlong([0, 1, 0]).translate(12, -halfD - 33, 23).color("#6fbb6c"),
);

const buildVolume = buildVolumeCage(buildVolumeW, buildVolumeD, 92, bedY, 31);
const toolheadGuide = box(34, 2, 2).translate(headX, -halfD + 12, gantryZ - 7).color(colors.guide);
const rearMotors = union(
  box(14, 10, 10).translate(-halfW + 26, halfD - 22, 22).color("#17191d"),
  box(14, 10, 10).translate(halfW - 26, halfD - 22, 22).color("#17191d"),
);
const details = union(toolheadGuide, rearMotors, buildVolume);

const printer = union(
  ...frameParts,
  heatedPlate,
  bedPlate,
  bedCarriage,
  bedRails,
  zMotion,
  gantry,
  carriageBody,
  fan,
  hotend,
  nozzle,
  heatsink,
  belts,
  spool,
  filamentPath,
  electronics,
  details,
);

const bb = printer.boundingBox();
verify.lessThan("overall width < 260 mm", bb.max[0] - bb.min[0], 260);
verify.lessThan("overall depth < 230 mm", bb.max[1] - bb.min[1], 230);
verify.lessThan("overall height < 260 mm", bb.max[2] - bb.min[2], 260);
verify.greaterThan("gantry stays above print bed", gantryZ, 65);
verify.lessThan("head stays inside frame", Math.abs(headX), halfW - 16);

return {
  frame: union(...frameParts),
  "orange heater plate": heatedPlate,
  "blue print bed": bedPlate,
  "bed carriage and rails": union(bedCarriage, bedRails),
  "cyan build volume cage": buildVolume,
  "z motion system": zMotion,
  "x gantry": gantry,
  "print head": union(carriageBody, fan, hotend, nozzle, heatsink),
  belts,
  "red spool": spool,
  filament: filamentPath,
  "front electronics": electronics,
  "small details": union(toolheadGuide, rearMotors),
};
