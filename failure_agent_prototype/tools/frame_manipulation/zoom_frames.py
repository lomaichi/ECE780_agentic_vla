import cv2, numpy as np, sys
# usage: crop.py folder steps x0,y0,x1,y1 scale out
d, steps, box, sc, out = sys.argv[1], sys.argv[2].split(','), [int(v) for v in sys.argv[3].split(',')], float(sys.argv[4]), sys.argv[5]
x0, y0, x1, y1 = box
ims = []
for s in steps:
    im = cv2.imread(f'{d}/frames/step{int(s):04d}.png')[y0:y1, x0:x1]
    im = cv2.resize(im, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
    cv2.putText(im, f'step {s}', (5, 22), 0, 0.7, (0, 255, 255), 2)
    ims.append(im)
cv2.imwrite(out, np.hstack(ims))
