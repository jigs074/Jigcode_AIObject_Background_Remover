import cv2
# library for image and video processing 
import numpy as np
#imports numpy for handling arrays and numerical operations
import torch
#imports pytorch a deep learning framework
import torchvision
#library to handle image transformation and pretrained models
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
#imports DeepLabV3 model and pretrained weights
import os


class BackgroundRemover:
    def __init__(self):
        """
        Initialize the background remover with a pre-trained segmentation model.
        """
        # Check if GPU is available, otherwise use CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load pre-trained DeepLabV3 model
        self.model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT).to(self.device)
        self.model.eval() 
        #setting up the model to evaluation mode (no training, just inference)
            
        # Set up preprocessing transforms
        self.preprocess = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def get_person_mask(self, image):
        """
        Generate a segmentation mask that identifies people in the image.
        
        Args:
            image (numpy.ndarray): Input image (BGR format from OpenCV)
                                  
        Returns:
            numpy.ndarray: Binary mask where 1 indicates pixels to keep (person)
        """
        # Convert BGR to RGB (OpenCV uses BGR, but our model expects RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Preprocess image for the model and adds a batch dimension
        input_tensor = self.preprocess(image_rgb).unsqueeze(0).to(self.device)
        # initially the format would be {C,H,W} after unsqueezing it we are adding a batch dimensions to it. for example if we have 5 images in the batch then it will be {5,C,H,W}
        # {C,H,W} -> UNSQUEEZE -> {10,C,H,W}
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
        # Get segmentation map
        segmentation_map = torch.argmax(output, dim=0).cpu().numpy()
    
        # Create binary mask for person class (15)
        mask = (segmentation_map == 15).astype(np.uint8)
        return mask
    
    def remove_background(self, image, bg_color=None, use_checkerboard=True):
        """
        Remove the background from an image, keeping only the person.
        
        Args:
            image (numpy.ndarray): Input image (BGR format from OpenCV)
            bg_color (tuple): BGR color to use for background if not using checkerboard
            use_checkerboard (bool): Whether to use a checkerboard pattern for background
            
        Returns:
            numpy.ndarray: Image with background replaced
        """
        # Get person mask
        person_mask = self.get_person_mask(image)
        # FOR PERSON it is going to return 1 and for everything else it will return 0
        # Create a 3-channel mask for blending 
        mask_3channel = np.stack([person_mask] * 3, axis=2)
        
        # Create background
        if use_checkerboard:
            # Create a checkerboard pattern
            checker_size = 20
            h, w = image.shape[:2]
            background = np.zeros((h, w, 3), dtype=np.uint8)
            for i in range(0, h, checker_size):
                for j in range(0, w, checker_size):
                    if (i//checker_size + j//checker_size) % 2 == 0:
                        background[i:i+checker_size, j:j+checker_size] = [200, 200, 200]  # Light gray
                    else:
                        background[i:i+checker_size, j:j+checker_size] = [100, 100, 100]  # Dark gray
        else:
            # Use solid color
            if bg_color is None:
                bg_color = (0, 255, 0)  # Green by default
            background = np.ones_like(image) * np.array(bg_color, dtype=np.uint8)
        
        # Blend the foreground (person) with the background
        result = image * mask_3channel + background * (1 - mask_3channel)
        # See the value of mask_3_channel is going to be 0 for background and 1 for person. result = image + background
        # value of the mask channel is 1 for person and 0 for background , image + background(1-0) = image+background
        return result.astype(np.uint8)
    
    def process_single_frame(self, input_path, output_path, frame_number=0, bg_color=None, use_checkerboard=True):
        """
        Process a single frame from a video and create an output with background removed.
        
        Args:
            input_path (str): Path to input video
            output_path (str): Path to save output video
            frame_number (int): Which frame to extract and process (default is the first frame)
            bg_color (tuple): BGR color to use for background if not using checkerboard
            use_checkerboard (bool): Whether to use a checkerboard pattern for background
        """
        print(f"Processing frame #{frame_number} from {input_path}")
        
        # Open the video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video at {input_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        # Skip to the desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError(f"Could not read frame {frame_number} from video")
        
        print("Processing frame...")
        
        # Remove background
        processed_frame = self.remove_background(frame, bg_color, use_checkerboard)
        
        # Save as MP4 video (short clip)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Write the processed frame to output video (multiple times to create a short video)
        for _ in range(int(fps * 3)):  # Create a 3-second video
            out.write(processed_frame)
        
        out.release()
        print(f"Video saved to {output_path}")
        
        # Also save as PNG with transparency 
        png_path = output_path.replace('.mp4', '.png')
        result_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        person_mask = self.get_person_mask(frame)
        result_bgra[:, :, 3] = person_mask * 255
        cv2.imwrite(png_path, result_bgra)
        print(f"Transparent PNG saved to {png_path}")
        
        # Release resources
        cap.release()
        
        return processed_frame


# Example usage
if __name__ == "__main__":
    # Initialize the background remover
    remover = BackgroundRemover()
    
    # Example 1: Green screen background
    remover.process_single_frame(
        input_path="input_video.mp4",  # Replace with your video file
        output_path="output_green_bg.mp4",
        bg_color=(0, 255, 0),  # Green background (BGR format)
        use_checkerboard=False,
        frame_number=0  # Process the first frame
    )
    
    # Example 2: Checkerboard background
    remover.process_single_frame(
        input_path="input_video.mp4",  # Replace with your video file
        output_path="output_checker_bg.mp4",
        use_checkerboard=True,
        frame_number=0  # Process the first frame
    )


