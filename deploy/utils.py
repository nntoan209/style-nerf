from infer_upst import generate_image_upst, generate_video_upst
from infer_stylerf import generate_image_stylerf, generate_video_stylerf
            
    
def generate_video(framework, scene, style_image):
    if framework == "UPST-NeRF":
        return generate_video_upst(scene, style_image)
    elif framework == "StyleRF":
        return generate_video_stylerf(scene, style_image)
    
def generate_image(framework, scene, style_image, theta, phi):
    if framework == "UPST-NeRF":
        return generate_image_upst(scene, style_image, theta, phi)
    elif framework == "StyleRF":
        return generate_image_stylerf(scene, style_image, theta, phi)
