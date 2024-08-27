from segment_anything import SamPredictor, sam_model_registry

model_type = "vit_h"  # Tipo do modelo SAM
sam = sam_model_registry[model_type](checkpoint='sam_vit_h_4b8939.pth')
predictor = SamPredictor(sam)

for image_path in image_dataset:
    image = load_image(image_path)
    predictor.set_image(image)
    masks, _, _ = predictor.predict()  # Previsão de máscaras
    # Processar e avaliar as máscaras
