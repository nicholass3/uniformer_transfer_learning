# import os
# import csv
#
# # Initialize empty list to hold the rows
# rows = []
#
# # Set the root directory where the UCF50 dataset is located
# root_dir = 'UCF50'  # Replace this with the actual path to your UCF50 dataset
#
# # Loop through the subfolders (labels) in the root directory
# for label in os.listdir(root_dir):
#     # Create the full path to the subfolder
#     subfolder_path = os.path.join(root_dir, label)
#
#     # Make sure we're only looking at directories
#     if os.path.isdir(subfolder_path):
#         # Loop through the video files in the subfolder
#         for video_name in os.listdir(subfolder_path):
#             # Create a row for this video with its name and label
#             row = [video_name, label]
#
#             # Add the row to the list of rows
#             rows.append(row)
#
# # Write the rows to a CSV file
# csv_file_path = 'video_labels.csv'  # You can change this to your desired output file path
# with open(csv_file_path, 'w', newline='') as csvfile:
#     csvwriter = csv.writer(csvfile)
#
#     # Write the header row
#     csvwriter.writerow(['Video Name', 'Label'])
#
#     # Write the data rows
#     csvwriter.writerows(rows)
#
# print(f"CSV file has been created at {csv_file_path}")

# import csv
#
# # Read the existing CSV file
# with open('ucf101_features.csv', 'r') as infile:
#     csvreader = csv.reader(infile)
#     header = next(csvreader)  # Skip header row
#     rows = [row for row in csvreader]
#
# # Update the video paths to only include the video name
# updated_rows = []
# for row in rows:
#     full_path = row[0]
#     video_name = full_path.split('/')[-1]  # Assuming the path uses forward slashes
#     updated_rows.append([video_name, row[1]])
#
# # Write the updated rows back to the CSV file (or a new one)
# with open('updated_ucf101_features.csv', 'w', newline='') as outfile:
#     csvwriter = csv.writer(outfile)
#     csvwriter.writerow(header)  # Write the header row
#     csvwriter.writerows(updated_rows)
#
# print(f"CSV file has been updated and saved as updated_video_labels.csv")

# import csv
#
# # Step 1: Load UCF101 labels
# ucf101_labels = {}
# with open('updated_ucf101_features.csv', 'r') as infile:  # Replace with your actual UCF101 csv path
#     csvreader = csv.reader(infile)
#     next(csvreader)  # Skip the header row
#     for row in csvreader:
#         video_name, label_number = row
#         ucf101_labels[video_name] = label_number
#
# # Step 2: Update UCF50 labels with corresponding UCF101 label numbers
# updated_ucf50_rows = []
# with open('ucf50_labels.csv', 'r') as infile:  # Replace with your actual UCF50 csv path
#     csvreader = csv.reader(infile)
#     header = next(csvreader)  # Skip header row
#     updated_ucf50_rows.append(header + ['label_number'])  # Add a new column header
#
#     for row in csvreader:
#         video_name, label_name = row
#         label_number = ucf101_labels.get(video_name,
#                                          'N/A')  # Use 'N/A' if the video_name doesn't exist in UCF101 labels
#         updated_row = row + [label_number]
#         updated_ucf50_rows.append(updated_row)
#
# # Step 3: Write the updated UCF50 rows to a new CSV file
# with open('updated_ucf50_video_labels.csv', 'w', newline='') as outfile:  # You can change the output file name
#     csvwriter = csv.writer(outfile)
#     csvwriter.writerows(updated_ucf50_rows)
#
# print("Updated UCF50 labels with UCF101 label numbers.")

# import os

# # Set the directory you want to start from
# directory_path = "UCF50/PullUps"

# # Loop through each file in the directory
# for filename in os.listdir(directory_path):
#     if "v_Pullup_g" in filename:  # Check if 'v_Pullup_g' is in the filename
#         new_filename = filename.replace("Pullup", "Pullups")  # Replace 'Pullup' with 'Pullups'
#         original_file_path = os.path.join(directory_path, filename)
#         new_file_path = os.path.join(directory_path, new_filename)
#
#         # Rename the file
#         os.rename(original_file_path, new_file_path)