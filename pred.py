from ultralytics import YOLO
from PIL import Image

model = YOLO("dev_40epoch.pt")

results = model.predict("test_5.png")


print(results)
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image