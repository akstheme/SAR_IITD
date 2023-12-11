from ultralytics import YOLO
import os
from ultralytics import settings
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import Counter
#settings.reset()
# Update a setting
#settings.update({'runs_dir': 'E:/IITD_ResearchWork/B_Reporter/detective/code/My_projects/'})

# Get user input for the subject
user_input = input("Enter the subject: ")

# Update the output directories based on user input
Person_Dir = f"{user_input}/Person"
Face_Dir = f"{user_input}/Face"
Fire_Dir = f"{user_input}/Fire"
Weapon_Dir = f"{user_input}/Weapon"

model_P = YOLO("E:/IITD_ResearchWork/B_Reporter/detective/code/My_projects/models/person.pt")
model_FC = YOLO("E:/IITD_ResearchWork/B_Reporter/detective/code/My_projects/models/human_face.pt")
model_FR = YOLO("E:/IITD_ResearchWork/B_Reporter/detective/code/My_projects/models/fire.pt")
model_W = YOLO("E:/IITD_ResearchWork/B_Reporter/detective/code/My_projects/models/weapon.pt")
model_FER = load_model("E:/IITD_ResearchWork/B_Reporter/detective/code/My_projects/fer_model_best.h5")

source = "E:/IITD_ResearchWork/B_Reporter/detective/code/My_projects/Test/h.jpg"

#model.predict(source, save=True, save_crop=True, classes=None, conf=0.5)
#res1 = model_P.predict(source, save=True, save_crop=True, classes=None, conf=0.5, save_conf=True, save_txt=True, save_json=True) #for person
#res2 = model_FC.predict(source, save=True, save_crop=True, classes=None, conf=0.5, save_conf=True, save_txt=True, save_json=True) # for face
#res3 = model_FR.predict(source, save=True, save_crop=True, classes=None, conf=0.5, save_conf=True, save_txt=True, save_json=True) # for fire
#res4 = model_W.predict(source, save=True, save_crop=True, classes=None, conf=0.5, save_conf=True, save_txt=True, save_json=True) # for weapons

# res1 = model_P(source, save=True, save_crop=True, classes=0, conf=0.5, save_conf=True, save_txt=True, project="detect/Person", name="inference", exist_ok=True) #for person
# res2 = model_FC(source, save=True, save_crop=True, classes=None, conf=0.5, save_conf=True, save_txt=True, project="detect/Face", name="inference", exist_ok=True) # for face
# res3 = model_FR(source, save=True, save_crop=True, classes=None, conf=0.5, save_conf=True, save_txt=True, project="detect/Fire", name="inference", exist_ok=True) # for fire
# res4 = model_W(source, save=True, save_crop=True, classes=None, conf=0.5, save_conf=True, save_txt=True, project="detect/Weapon", name="inference", exist_ok=True) # for weapons


res1 = model_P(source, save=True, save_crop=True, classes=0, conf=0.5, save_conf=True, save_txt=True, project=Person_Dir, name="inference", exist_ok=True) #for person
res2 = model_FC(source, save=True, save_crop=True, classes=None, conf=0.5, save_conf=True, save_txt=True, project=Face_Dir, name="inference", exist_ok=True) # for face
res3 = model_FR(source, save=True, save_crop=True, classes=None, conf=0.5, save_conf=True, save_txt=True, project=Fire_Dir, name="inference", exist_ok=True) # for fire
res4 = model_W(source, save=True, save_crop=True, classes=None, conf=0.5, save_conf=True, save_txt=True, project=Weapon_Dir, name="inference", exist_ok=True) # for weapons



# View results
count_persons = 0
for r1 in res1:
    a = r1.boxes  # print the Boxes object containing the detection bounding boxes
predicted_Person = a.cls.tolist()
confidence_scoresP = a.conf.tolist()
bounding_box_coordinatesP = a.xyxy.tolist()

for r2 in res2:
    b = r2.boxes
predicted_Face = b.cls.tolist()
confidence_scoresFC = b.conf.tolist()
bounding_box_coordinatesFC = b.xyxy.tolist()   
    
for r3 in res3:
    c = r3.boxes
predicted_Fire = c.cls.tolist()
confidence_scoresFR = c.conf.tolist()
bounding_box_coordinatesFR = c.xyxy.tolist()

for r4 in res4:
    d = r4.boxes
predicted_W = d.cls.tolist()
confidence_scoresW = d.conf.tolist()
bounding_box_coordinatesW = d.xyxy.tolist()

# get the output directory 

# Function to get a list of all label files in the directory
def get_label_files(directory):
    label_files = [file for file in os.listdir(directory) if file.endswith('.txt')]
    return label_files

# Function to check if there are any label files in the directory
def has_label_files(directory):
    return any(file.endswith('.txt') for file in os.listdir(directory))

# Function to read and show the image if label files exist
def read_and_show_image(directory, directory1):
    predicted_classes_T = []  # List to store predicted classes
    predicted_labels_T = []
    if has_label_files(directory):
        image_files = [file for file in os.listdir(directory1) if file.lower().endswith('.jpg')]
        if image_files:
            for image_file in image_files:
                image_path = os.path.join(directory1, image_file)
                #image = cv2.imread(image_path)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale
                image = cv2.resize(image, (48, 48))  # Resize to match the model's input size
                image = image / 255.0  # Normalize pixel values to [0, 1]

                # Reshape the image to match the input shape expected by the model
                image = np.reshape(image, (1, 48, 48, 1))  # Adjust dimensions based on your model's input shape

                # Make predictions
                predictions = model_FER.predict(image)

                # Post-processing (optional)
                predicted_class = np.argmax(predictions)
                predicted_classes_T.append(predicted_class)
                predicted_label = emotion_labels[predicted_class]
                predicted_labels_T.append(predicted_label)
                #print("Predicted class:", predicted_label)
        else:
            print('No JPG images found in the directory.')

    else:
        print('No detection')
    return predicted_classes_T, predicted_labels_T


########################################## Riskometer  ######################################################
def calculate_risk_factor(no_of_person, no_of_weapons, no_of_faces, no_of_fire, emotion_types):
    # Step 1: Find what is present in the input image
    has_person = no_of_person > 0
    has_weapons = no_of_weapons > 0
    has_fire = no_of_fire > 0
    has_faces = no_of_faces > 0
    # is_angry = emotion_type == "anger"
    # is_fearful = emotion_type == "fear"

    # Step 2: Create cases
    #w_fire = 0
    #w_weapons = 0
    w_person = 0
    w_emotion = 0

    if has_person:
        # Case IV: Only person is present
        if no_of_person < 5:
            RF = 0
            w_person = 0
        elif 5 <= no_of_person < 10:
            RF = 0.1
            w_person = 0.1
        else:
            RF = 0.2
            w_person=0.2

    # Step 3: Consider emotions
# Step 3: Consider emotions
    if has_faces:
        #num_sad = emotion_types.count('sad')
        num_anger = emotion_types.count('anger')
        num_fear = emotion_types.count('fear')
        #num_neutral = emotion_types.count('neutral')

        if num_anger > 0 and num_fear == 0:
            # Case I: Anger
            w_emotion = 0.2
            Anger_cnt = 1
        elif num_anger == 0 and num_fear > 0:
            # Case II: Fear
            w_emotion = 0.2
            fear_cnt = 1
        elif num_anger > 0 and num_fear > 0:
            # Case V: Both anger and fear
            w_emotion = 0.3
            AF_cnt = 1
    # Step 4: Calculate Risk Factor (RF)
    if has_fire and not has_weapons and not has_person:
        # Case I: Only fire is there
        RF = 0.3
    elif not has_fire and has_weapons and not has_person:
        # Case II: Only weapon is there
        RF = 0.2
    elif has_fire and has_weapons and not has_person:
        # Case III: Fire and weapons are there
        RF = 0.2 + 0.2
    elif has_person and not has_weapons and not has_fire:
        # Case IV: Only person is there
        RF = w_person + w_emotion
    elif has_person and has_fire and not has_weapons and not has_faces:
        # Case V: Person and fire are there
        RF = w_person + w_emotion + 0.3
    elif has_person and has_weapons and not has_fire and not has_faces:
        # Case VI: Person and weapon are there
        RF = w_person + w_emotion + 0.2 + 0.3
    elif has_person and has_weapons and has_fire and not has_faces:
        # Case VII: Person, weapon, and fire are there
        RF = w_person + w_emotion + 0.4 + 0.2
    elif has_faces:
        if Anger_cnt > 0 and has_weapons:
            RF = 1
        elif fear_cnt > 0 and has_weapons:
            RF = 1
        elif AF_cnt > 0 and has_weapons:
            RF = 1
    RF = min(RF,1)

    return RF

 
directory_path = f"{user_input}/Face/inference/labels/" # for label detection (Detection or not)
directory_path1 = f"{user_input}/Face/inference/"  #can be use for dispaly image
directory_path2 = f"{user_input}/Face/inference/crops/human_face/" #for emotion recognition
emotion_labels = {
    0: 'anger',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral'
}

predicted_classes, predicted_labels = read_and_show_image(directory_path, directory_path2)
print(predicted_classes)
print(predicted_labels)
#
# Calculate the frequency of each value
frequency = Counter(predicted_classes)

# Display the frequency of each class
for value, count in frequency.items():
    print(f'Class {value}: {count} occurrences')




print('No of Persons:', len(confidence_scoresP))#0.1
print('No of Faces:', len(confidence_scoresFC))#0.3
print('No of Fire Objects:', len(confidence_scoresFR))#0.3
print('No of Weapons:', len(confidence_scoresW))#risk_weight=0.3

# Example usage:
no_of_person = len(confidence_scoresP)#8
no_of_weapons = len(confidence_scoresW)#1
no_of_faces = len(confidence_scoresFC)#1
no_of_fire = len(confidence_scoresFR)#1
emotion_type = predicted_labels

risk_factor = calculate_risk_factor(no_of_person, no_of_weapons, no_of_faces, no_of_fire, emotion_type)
print(f"Risk Factor#################: {risk_factor}")

