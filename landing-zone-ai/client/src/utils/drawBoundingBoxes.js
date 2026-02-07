export const drawBoundingBoxes = (canvas, boxes) => {
  const ctx = canvas.getContext('2d');
  ctx.strokeStyle = 'red';
  ctx.lineWidth = 2;

  boxes.forEach(box => {
    const { x, y, width, height, label } = box;
    ctx.strokeRect(x, y, width, height);
    if (label) {
        ctx.fillStyle = 'red';
        ctx.font = '12px Arial';
        ctx.fillText(label, x, y - 5);
    }
  });
};
