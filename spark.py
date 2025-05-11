from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, regexp_extract, rand
from pyspark.sql.types import ArrayType, FloatType
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import random
import torch.nn.functional as F
import os
from deeplab import MobileNetV3_DeepLab


# Start Spark session 
# Runs in local mode with 16GB memory for driver and executor, for 24 executors. 
# Timeouts are set to ridiculously high values to avoid any issues with long-running tasks.
spark = SparkSession.builder \
    .appName("531") \
    .master("local[*]") \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.memory", "16g") \
    .config("spark.executor.heartbeatInterval", "100000s") \
    .config("spark.network.timeout", "1000000s") \
    .config("spark.python.task.timeout", "1000000") \
    .getOrCreate()

# Load images from directories
frames_dir = "data/frames"
masks_dir = "data/masks"

frames_df = spark.read.format("image").load(frames_dir)
masks_df = spark.read.format("image").load(masks_dir)

# Extract image file names as ID. Frames and masks match on file names / ID. 
frames_df = frames_df.withColumn("id", regexp_extract("image.origin", r"([^/\\\\]+)$", 1))
masks_df = masks_df.withColumn("id", regexp_extract("image.origin", r"([^/\\\\]+)$", 1))

# Join on ID first to preserve alignment
joined_df = frames_df.alias("f").join(masks_df.alias("m"), on="id") \
    .select(
        "id", 
        col("f.image.data").alias("frame_data"), 
        col("f.image.height").alias("frame_height"), 
        col("f.image.width").alias("frame_width"), 
        col("f.image.nChannels").alias("frame_channels"),
        col("m.image.data").alias("mask_data")
    )

# Repartition the DataFrame to 1000 partitions for better parallelism
joined_df = joined_df.repartition(1000)

# Shuffle the DataFrame globally
shuffled_df = joined_df.orderBy(rand())
rdd_with_index = shuffled_df.rdd.zipWithIndex()

# Convert back to DataFrame, adding the 'row_num' column. This corresponds to the original row number in the shuffled DataFrame.
shuffled_df = rdd_with_index.map(lambda x: x[0] + (x[1],)).toDF(shuffled_df.columns + ['row_num'])

# Cache the DataFrame to speed up subsequent operations. We will only be reading from shuffled_df from here on
shuffled_df = shuffled_df.cache()

# UDF definition & registration
# These change the binary data to numpy arrays, and then to lists.
def binary_to_image_array(data, height, width, nChannels):
    if data is None:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    if len(arr) != height * width * nChannels:
        return None
    return arr.reshape((height, width, nChannels)).tolist()  # Shape: H x W x C

def binary_mask_to_array(data, height, width):
    if data is None:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    return arr.reshape((height, width)).tolist()  # Shape: H x W

def array_to_list(arr):
    return np.array(arr, dtype=np.float32).flatten().tolist()

reshape_udf = udf(binary_to_image_array, ArrayType(FloatType()))
flatten_mask_udf = udf(binary_mask_to_array, ArrayType(FloatType()))
vectorize_udf = udf(array_to_list, ArrayType(FloatType()))

# Function to extract a batch
# Done by filtering our cached shuffled_df based on assigned 'row_num'.
def get_batch(df, batch_size, batch_idx):
    start = batch_idx * batch_size + 1
    end = start + batch_size
    return df.filter((col("row_num") >= start) & (col("row_num") < end))

###############################################################################################################################

# Preprocessing transform for data augmentation
# This class is written to randomly apply transformations to frame-mask PAIRS.
# i.e. if one frame is flipped, the corresponding mask is also flipped.
class RandomTransform:
    def __init__(self, resize=(512, 512)):
        self.resize = resize
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, frame, mask):
        # Convert numpy or tensor to PIL
        
        frame = self.to_pil(frame)
        mask = self.to_pil(mask)

        # Apply same random horizontal flip
        if random.random() > 0.5:
            frame = TF.hflip(frame)
            mask = TF.hflip(mask)
        
        # Apply random vertical flip
        if random.random() > 0.5:
            frame = TF.vflip(frame)
            mask = TF.vflip(mask)

        # Apply same random rotation
        if random.random() > 0.5:
            angle = random.uniform(-30, 30)
            frame = TF.rotate(frame, angle)
            mask = TF.rotate(mask, angle)
        
        # Apply some random translation    
        if random.random() > 0.5:
            translate = (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1))
            frame = TF.affine(frame, angle=0, translate=translate, scale=1.0, shear=0)
            mask = TF.affine(mask, angle=0, translate=translate, scale=1.0, shear=0)

        # Resize both to 512x512 (or whatever size you want)
        frame = TF.resize(frame, self.resize)
        mask = TF.resize(mask, self.resize)

        # Convert back to tensor to feed into the model
        frame = self.to_tensor(frame)
        frame = self.normalize(frame) 
        mask = (self.to_tensor(mask) > 0).float()
        
        return frame, mask


# Custom Dataset to convert PySpark DataFrame to PyTorch tensors
class ImageSegmentationDataset(Dataset):
    def __init__(self, row_list):
        self.data = row_list
        self.transform = RandomTransform()
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx].asDict()
        
        # Ensure proper reshaping of frame (3, 512, 512)
        frame = np.array(row['frame_vector'], dtype=np.float32).reshape(512, 512, 3)
        
        # Mask reshaping (1, 512, 512)
        mask = np.array(row['mask_vector'], dtype=np.float32).reshape(512, 512, 1)

        frame, mask = self.transform(frame, mask)  # Correctly transformed to tensor (C x H x W)

        return frame, mask


####################################################################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
model = MobileNetV3_DeepLab(num_classes=1).to(device)  # Move model to GPU if available

# Binary Cross-Entropy loss with logits for pixel-wise binary classification
criterion = nn.BCEWithLogitsLoss()
criterion = criterion.to(device)  

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

#####################################################################################################################

# Function to visualize the transformed frame and mask. Used for debugging.
def showTransform(frames, masks, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # Unnormalize the frame
    frame = frames[0].cpu().clone()
    mean = torch.tensor(mean, dtype=frame.dtype, device=frame.device).view(3, 1, 1)
    std = torch.tensor(std, dtype=frame.dtype, device=frame.device).view(3, 1, 1)
    frame = frame * std + mean

    # Convert to numpy for display
    frame_np = frame.permute(1, 2, 0).numpy()

    # Mask (already in binary form)
    mask_np = masks[0].squeeze().cpu().numpy() * 255

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(np.clip(frame_np, 0, 1))  # Clipping for safe visualization
    axs[0].set_title("Frame")
    axs[0].axis("off")

    axs[1].imshow(mask_np, cmap='gray')
    axs[1].set_title("Mask")
    axs[1].axis("off")

    plt.show()



####################################################################################################################################

os.makedirs("models", exist_ok=True)

epochs = 50
batch_size = 8
patience = 3
best_loss = float("inf")
early_stop_counter = 0

total_rows = shuffled_df.count()
num_batches = total_rows // batch_size + (1 if total_rows % batch_size != 0 else 0)

# Training loop
# For each epoch, a batch_size of data is loaded from the shuffled DataFrame through Spark.
# The Dataloader minibatches above data for the model. This is done per epoch (all available data)
# Early stopping is implemented based on the average training loss per epoch.
for epoch in range(epochs):
    
    print(f"Epoch {epoch + 1}/{epochs}")
    epoch_loss = 0.0
    epoch_correct = 0
    epoch_total = 0
    
    for batch_idx in range(num_batches):
        
        batch_df = get_batch(shuffled_df, batch_size, batch_idx)

        processed_batch = batch_df \
            .withColumn("frame", reshape_udf("frame_data", "frame_height", "frame_width", "frame_channels")) \
            .withColumn("mask", flatten_mask_udf("mask_data", "frame_height", "frame_width")) \
            .withColumn("frame_vector", vectorize_udf("frame")) \
            .withColumn("mask_vector", vectorize_udf("mask")) \
            .select("id", "frame_vector", "mask_vector")
        
        # Prepare the dataset for PyTorch
        dataset = ImageSegmentationDataset(processed_batch.collect())

        # Create a DataLoader to handle batching
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True, pin_memory=True, num_workers=0)
        

        for i, (frames, masks) in enumerate(dataloader):
            # Move data to the GPU (if available)
            frames = frames.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
        
            # showTransform(frames, masks)
            
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass: Predict the segmentation mask
            outputs = model(frames)

            # Calculate the loss
            loss = criterion(outputs, masks)

            # Backward pass: Compute gradients
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update the model weights
            optimizer.step()
            
            # Accumulate loss
            epoch_loss += loss.item()

            # Compute batch accuracy
            with torch.no_grad():
                preds = torch.sigmoid(outputs)
                preds = (preds > 0.5).float()
                correct = (preds == masks).float().sum()
                total = masks.numel()
                acc = correct / total
                epoch_correct += correct.item()
                epoch_total += total

                print(f"Batch {batch_idx + 1}/{num_batches}, Minibatch {i + 1} — Loss: {loss.item():.4f}, Accuracy: {acc.item() * 100:.2f}%")
                
        checkpoint_path = f"models/model_epoch_{epoch + 1}_batch_{batch_idx + 1}.pth"
        torch.save(model.state_dict(), checkpoint_path)

    torch.cuda.empty_cache()
            
    avg_loss = epoch_loss / num_batches
    avg_acc = epoch_correct / epoch_total
    
    print(f"Epoch {epoch + 1} Summary — Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_acc * 100:.2f}%")

    # Early stopping
    if avg_loss < best_loss:
        best_loss = avg_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), f"models/best_model_epoch_{epoch + 1}.pth")
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break
        
####################################################################################################################

spark.stop()
