import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

def create_synthetic_dataset(output_dir='./test_data', samples_per_class=50):
    print("Creating synthetic ASL dataset for testing...")
    print("Note: This is for testing only. Use real dataset for actual training!")
    
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
               'del', 'nothing', 'space']
    
    os.makedirs(output_dir, exist_ok=True)
    
    colors = [
        (255, 200, 200), (200, 255, 200), (200, 200, 255),
        (255, 255, 200), (255, 200, 255), (200, 255, 255),
        (255, 220, 180), (220, 255, 180), (180, 220, 255)
    ]
    
    total_images = 0
    
    for class_name in classes:
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        for i in range(samples_per_class):
            img = Image.new('RGB', (200, 200), color=random.choice(colors))
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.truetype("arial.ttf", 80)
            except:
                font = ImageFont.load_default()
            
            text = class_name if class_name not in ['del', 'nothing', 'space'] else class_name[:3]
            
            noise = np.random.randint(0, 30, (200, 200, 3), dtype=np.uint8)
            img_array = np.array(img)
            img_array = np.clip(img_array.astype(np.int16) + noise - 15, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)
            draw = ImageDraw.Draw(img)
            
            x_offset = random.randint(-20, 20)
            y_offset = random.randint(-20, 20)
            
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (200 - text_width) // 2 + x_offset
            y = (200 - text_height) // 2 + y_offset
            
            text_color = (
                random.randint(0, 100),
                random.randint(0, 100),
                random.randint(0, 100)
            )
            draw.text((x, y), text, fill=text_color, font=font)
            
            for _ in range(random.randint(20, 50)):
                x1 = random.randint(0, 200)
                y1 = random.randint(0, 200)
                x2 = x1 + random.randint(-30, 30)
                y2 = y1 + random.randint(-30, 30)
                color = tuple(random.randint(0, 255) for _ in range(3))
                draw.line((x1, y1, x2, y2), fill=color, width=1)
            
            img_path = os.path.join(class_dir, f'{class_name}_{i:04d}.jpg')
            img.save(img_path)
            total_images += 1
        
        print(f"✓ Created {samples_per_class} images for class '{class_name}'")
    
    print(f"\n✓ Total images created: {total_images}")
    print(f"✓ Dataset saved to: {output_dir}")
    print(f"\nTo train with this test data:")
    print(f"python train.py --data_dir {output_dir} --epochs 5 --batch_size 16")
    print("\n⚠ Warning: This synthetic data is for testing only!")
    print("For real training, use: python download_data.py --dataset alphabet")

def create_single_test_image(output_path='test_image.jpg', letter='A'):
    print(f"Creating single test image for letter '{letter}'...")
    
    img = Image.new('RGB', (200, 200), color=(220, 230, 240))
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 100)
    except:
        font = ImageFont.load_default()
    
    bbox = draw.textbbox((0, 0), letter, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (200 - text_width) // 2
    y = (200 - text_height) // 2
    
    draw.text((x, y), letter, fill=(50, 50, 50), font=font)
    
    img.save(output_path)
    print(f"✓ Test image saved to: {output_path}")
    print(f"You can test it with: Load Image in the app")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create synthetic test data for ASL model')
    parser.add_argument('--samples', type=int, default=50, help='Samples per class')
    parser.add_argument('--output_dir', type=str, default='./test_data', help='Output directory')
    parser.add_argument('--single_image', action='store_true', help='Create single test image')
    parser.add_argument('--letter', type=str, default='A', help='Letter for single image')
    args = parser.parse_args()
    
    if args.single_image:
        create_single_test_image(letter=args.letter)
    else:
        create_synthetic_dataset(args.output_dir, args.samples)