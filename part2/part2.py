from tensorflow.keras.models import load_model
import numpy as np

def get_model():
    model=load_model("model.h5")
    return model


def get_label(model,image):

    crop_shape = (81, 81)
    test_image=np.array(image)
    test_image = test_image.reshape([-1] + list(crop_shape) + [3])
    predictions = model.predict(test_image)
    predicted_label = np.argmax(predictions, axis=-1)
    return predicted_label