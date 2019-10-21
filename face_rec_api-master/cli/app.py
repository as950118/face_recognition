import requests

# Define testing endpoints
training_URL = "http://localhost:5000/train"
recognizing_URL = "http://localhost:5000/recognize"
detecting_URL = "http://localhost:5000/detect"

training_images = {
    "ben": './training_image/Ben.jpg',
    "biden": './training_image/biden.jpg',
    "emma": './training_image/emma.jpg',
    "liu": './training_image/liu.jpg',
    "obama": './training_image/obama.jpg',
    "pyy": './training_image/pyy.jpg',
    "taylor": './training_image/taylor.jpg',
    "trump": './training_image/trump.jpg',
    "wen": './training_image/wen.jpg'
}

testing_images = {
    "ben": './testing_image/benTest.jpg',
    "emma": './testing_image/emmaTest.jpg',
    "liu": './testing_image/liuTest.jpg',
    "two_people": './testing_image/two_people.jpg',
    "pyy": './testing_image/pyyTest.jpg',
    "taylor": './testing_image/taylorTest.jpg',
    "trump": './testing_image/trumpTest.jpg',
    "wen": './testing_image/wenTest.jpg'
}

def test_single_training_rec():
    training_files = {'image': open(training_images["ben"], 'rb')}
    testing_files = {'image': open(testing_images["ben"], 'rb')}

    # Send the training image
    r = requests.post(url = training_URL, files=training_files)

    if r.status_code == 404:
        print("Error: duplicated name or face")
        return

    data = r.json()
    print(data['msg'])

    # Send the testing image
    r = requests.post(url = recognizing_URL, files=testing_files)

    # Save the response image
    open('./result_image/ben_recognized.jpg', 'wb').write(r.content)
    return

def test_multiple_training_rec():
    training_files = [{'image': open(training_images["biden"], 'rb')}, {'image': open(training_images["obama"], 'rb')}]
    testing_files = {'image': open(testing_images["two_people"], 'rb')}

    # Send the training images
    for training_file in training_files:
        r = requests.post(url = training_URL, files=training_file)
        if r.status_code == 404:
            print("Error: duplicated name or face")
            return

        data = r.json()
        print(data['msg'])


    # Send the testing image
    r = requests.post(url = recognizing_URL, files=testing_files)

    # Save the response image
    open('./result_image/two_people_recognized.jpg', 'wb').write(r.content)
    return

def test_single_detection():
    testing_files = {'image': open(testing_images["trump"], 'rb')}

    # Send the testing image
    r = requests.post(url = detecting_URL, files=testing_files)

    # Save the response image
    open('./result_image/trump_detected.jpg', 'wb').write(r.content)
    return

if __name__ == "__main__":
    test_single_training_rec()
    test_multiple_training_rec()
    test_single_detection()