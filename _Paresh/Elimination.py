import json
import os
import glob


path = 'C:/Users/Paresh P/Videos/OpenCamera/FinalTraining/New Mirrored/Pushing/8_output'
with open('C:/Users/Paresh P/Videos/OpenCamera/FinalTraining/New Mirrored/Pushing/8_output/modified_files.txt', 'a') as log:

    for filename in glob.glob(os.path.join(path, '*.json')):

        with open(filename, 'r+') as f:
            print(filename)
            json_data = json.load(f)
            json_persons = json_data["people"]

            delFlag = []

            for i in range(len(json_persons)):

                skeleton = json_persons[i]["pose_keypoints"]
                zeroCount = 0
                for j in range(2,len(skeleton),3):

                    if skeleton[j] == 0 :
                        zeroCount +=1

                if zeroCount > 6 :
                    delFlag.append(i)

            for element in reversed(delFlag):
                json_persons.pop(element) #too many missing values
                log.write(filename+'\n')

            json_data["people"] = json_persons
            f.seek(0)
            f.write(json.dumps(json_data))
            f.truncate()
