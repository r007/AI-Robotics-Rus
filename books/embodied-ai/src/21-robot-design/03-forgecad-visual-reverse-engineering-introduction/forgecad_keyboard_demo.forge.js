// ForgeCAD keyboard reconstruction demo inspired by the keyboard scene in the
// reference video. This is a local, runnable teaching model rather than an
// extracted asset from the video.

const cols = Param.number("Columns", 12, { min: 8, max: 14, unit: "keys" });
const keyPitch = Param.number("Key Pitch", 15, { min: 12, max: 18, unit: "mm" });
const caseAngle = Param.number("Case Angle", 6, { min: 0, max: 12, unit: "deg" });
const knobOffset = Param.number("Knob Offset", 0, { min: -18, max: 18, unit: "mm" });
const accentKeyX = Param.number("Accent Key X", 2, { min: -4, max: 4, unit: "keys" });

const rows = 5;
const keySize = keyPitch - 2.6;
const boardW = cols * keyPitch + 34;
const boardD = rows * keyPitch + 42;
const baseH = 8;
const deckZ = baseH + 2;
const keyZ = deckZ + 2.2;

const colors = {
  shell: "#202225",
  bevel: "#34363a",
  deck: "#4a4d52",
  keyLight: "#e1d9bd",
  keyDark: "#464749",
  keyMid: "#9b988b",
  red: "#c94b55",
  stem: "#222426",
  knob: "#b7b0a0",
  metal: "#8b9096",
};

function keycap(x, y, w, d, color, height = 6) {
  const base = box(w, d, height).translate(x, y, keyZ).color(color);
  const top = box(w * 0.82, d * 0.82, 1.8).translate(x, y, keyZ + height).color(color);
  const stem = box(w * 0.38, d * 0.38, 2.4).translate(x, y, deckZ + 0.2).color(colors.stem);
  return union(stem, base, top);
}

function rowKeys(rowIndex, count, xShift, colorPattern) {
  const y = (rowIndex - (rows - 1) / 2) * keyPitch;
  const startX = -((count - 1) * keyPitch) / 2 + xShift;
  const keys = [];
  for (let i = 0; i < count; i += 1) {
    const x = startX + i * keyPitch;
    const color = colorPattern(i, rowIndex);
    keys.push(keycap(x, y, keySize, keySize, color));
  }
  return union(...keys);
}

const base = box(boardW, boardD, baseH)
  .translate(0, 0, 0)
  .color(colors.shell)
  .material({ metalness: 0.15, roughness: 0.42 });

const rearLift = box(boardW - 10, 18, 10)
  .translate(0, boardD / 2 - 12, 0)
  .color(colors.bevel);

const deck = box(boardW - 18, boardD - 20, 3)
  .translate(0, 2, deckZ)
  .color(colors.deck)
  .material({ metalness: 0.25, roughness: 0.35 });

const leftFunctionColumn = union(
  keycap(-boardW / 2 + 14, -keyPitch * 1.7, keySize, keySize, colors.keyDark),
  keycap(-boardW / 2 + 14, -keyPitch * 0.65, keySize, keySize, colors.keyDark),
  keycap(-boardW / 2 + 14, keyPitch * 0.4, keySize, keySize, colors.keyDark),
  keycap(-boardW / 2 + 14, keyPitch * 1.45, keySize, keySize, colors.keyDark),
);

const topRow = rowKeys(2, cols, 1.5, (i) => {
  if (i === Math.round(cols / 2 + accentKeyX)) return colors.red;
  return i < 2 || i > cols - 3 ? colors.keyDark : colors.keyLight;
});

const alphaRow1 = rowKeys(1, cols - 1, -2, (i) => (i % 5 === 0 ? colors.keyMid : colors.keyLight));
const alphaRow2 = rowKeys(0, cols - 1, 4, (i) => (i % 6 === 4 ? colors.keyMid : colors.keyLight));
const alphaRow3 = rowKeys(-1, cols - 2, -1, (i) => (i < 1 || i > cols - 5 ? colors.keyDark : colors.keyLight));

const spaceRow = union(
  keycap(-boardW * 0.33, -keyPitch * 2, keySize, keySize, colors.keyDark),
  keycap(-boardW * 0.22, -keyPitch * 2, keySize, keySize, colors.keyDark),
  keycap(0, -keyPitch * 2, keyPitch * 4.2, keySize, colors.keyLight, 5.5),
  keycap(boardW * 0.27, -keyPitch * 2, keySize, keySize, colors.keyDark),
  keycap(boardW * 0.38, -keyPitch * 2, keySize, keySize, colors.keyDark),
);

const trackPoint = cylinder(4, 3.2, 2.2, 32)
  .translate(keyPitch * 1.3, keyPitch * 0.12, keyZ + 6)
  .color(colors.red);

const rightCluster = union(
  keycap(boardW / 2 - 20, keyPitch * 1.2, keySize, keySize, colors.keyDark),
  keycap(boardW / 2 - 20, keyPitch * 0.15, keySize, keySize, colors.keyDark),
  keycap(boardW / 2 - 20, -keyPitch * 0.9, keySize, keySize, colors.keyDark),
  cylinder(8, 8.5, undefined, 48)
    .pointAlong([0, 1, 0])
    .translate(boardW / 2 - 20 + knobOffset, -keyPitch * 2.05, keyZ + 4)
    .color(colors.knob)
    .material({ metalness: 0.35, roughness: 0.32 }),
);

const ports = union(
  box(20, 2.4, 5).translate(-boardW * 0.18, boardD / 2 + 0.6, baseH + 4).color(colors.metal),
  box(12, 2.4, 5).translate(boardW * 0.05, boardD / 2 + 0.6, baseH + 4).color(colors.metal),
  box(9, 2.4, 5).translate(boardW * 0.22, boardD / 2 + 0.6, baseH + 4).color(colors.metal),
);

const frontChamferHint = box(boardW - 8, 6, 4)
  .translate(0, -boardD / 2 + 4, 1)
  .color("#151719");

const keyboard = union(
  base,
  rearLift,
  deck,
  leftFunctionColumn,
  topRow,
  alphaRow1,
  alphaRow2,
  alphaRow3,
  spaceRow,
  trackPoint,
  rightCluster,
  ports,
  frontChamferHint,
).rotateAroundAxis([1, 0, 0], -caseAngle, [0, 0, 0]);

const bb = keyboard.boundingBox();
verify.lessThan("keyboard width < 240 mm", bb.max[0] - bb.min[0], 240);
verify.lessThan("keyboard depth < 150 mm", bb.max[1] - bb.min[1], 150);
verify.lessThan("keyboard height < 45 mm", bb.max[2] - bb.min[2], 45);
verify.greaterThan("has at least 8 columns", cols, 7);

return {
  "keyboard case": union(base, rearLift, deck, frontChamferHint).rotateAroundAxis([1, 0, 0], -caseAngle, [0, 0, 0]),
  "cream keycaps": union(topRow, alphaRow1, alphaRow2, alphaRow3, spaceRow).rotateAroundAxis([1, 0, 0], -caseAngle, [0, 0, 0]),
  "dark function keys": union(leftFunctionColumn, rightCluster).rotateAroundAxis([1, 0, 0], -caseAngle, [0, 0, 0]),
  "red accent controls": trackPoint.rotateAroundAxis([1, 0, 0], -caseAngle, [0, 0, 0]),
  "rear ports": ports.rotateAroundAxis([1, 0, 0], -caseAngle, [0, 0, 0]),
};
