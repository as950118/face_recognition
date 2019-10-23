from flask import Flask, request, jsonify, send_file, abort
from PIL import Image, ImageDraw
from io import BytesIO
import face_recognition
import numpy as np
import cv2
import os
from cProfile import Profile
from pstats import Stats

# Init app
app = Flask(__name__)

MAX_FACES = 2
index = 0
known_face_encodings = []
known_face_names = []


@app.route('/train', methods=['POST'])
def post_training():
    global index
    global MAX_FACES
    global known_face_encodings
    global known_face_names

    training_img = request.files['image']
    if training_img:
        training_filename = training_img.filename

        # If the face is already learned or has a duplicate name, report an error
        if training_filename[:-4] in known_face_names:
            abort(404)

        # Read the file in bytes, reduce IO operation
        img_data = BytesIO(training_img.read())

        # Load a sample picture and learn how to recognize it
        training_image = face_recognition.load_image_file(img_data)
        training_encoding = face_recognition.face_encodings(training_image, num_jitters=100)[0]

        # When the pool is full, remove the first trained data
        if index >= MAX_FACES:
            known_face_encodings.pop(0)
            known_face_names.pop(0)
        else:
            index += 1

        # Create arrays of known face encodings and their names
        known_face_encodings.append(training_encoding)
        known_face_names.append(training_filename[:-4])

        msg = 'Learned encoding for ' + \
            str(len(known_face_encodings)) + \
            ' images, including ' + ' '.join(known_face_names) + '.'

    return jsonify({
        'msg': msg
    })


@app.route('/recognize', methods=['POST'])
def post_recognizing():
    global known_face_encodings
    global known_face_names

    if len(known_face_encodings) <= 0:
        return jsonify({
            'msg': 'Error: No face learned in the database.'
        })

    unknown_img = request.files['image']
    if unknown_img:
        # Read the file in bytes, reduce IO operation
        img_data = BytesIO(unknown_img.read())
        img = Image.open(img_data).convert('RGB')
        img = np.array(img)

        # Load the unknown picture and find all the faces and face encodings in the unknown image
        #unknown_face_image = face_recognition.load_image_file(img_data)
        unknown_face_image = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

        face_locations = face_recognition.face_locations(
            unknown_face_image, number_of_times_to_upsample=2)
        face_encodings = face_recognition.face_encodings(
            unknown_face_image, face_locations, num_jitters=10)

        pil_image = Image.fromarray(unknown_face_image)
        draw = ImageDraw.Draw(pil_image)

        # Loop through each face found in the unknown image
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding, tolerance=0.6)
            # display(matches)

            name = "Unknown"

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            # Draw a box around the face using the Pillow module
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

            # Draw a label with a name below the face
            text_width, text_height = draw.textsize(name)
            draw.rectangle(((left, bottom - text_height - 10),
                            (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
            draw.text((left + 6, bottom - text_height - 5),
                      name, fill=(255, 255, 255, 255))

        # Remove the drawing library from memory as per the Pillow docs
        del draw

    return serve_pil_image(pil_image)


@app.route('/detect', methods=['POST'])
def post_detection():
    unknown_img = request.files['image']
    if unknown_img:
        # Read the file in bytes, reduce IO operation
        img_data = BytesIO(unknown_img.read())
        img = Image.open(img_data).convert('RGB')
        img = np.array(img)

        # Load the unknown picture and find all the faces and face encodings in the unknown image
        #unknown_face_image = face_recognition.load_image_file(img_data)
        unknown_face_image = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

        face_locations = face_recognition.face_locations(
            unknown_face_image, number_of_times_to_upsample=2)
        face_encodings = face_recognition.face_encodings(
            unknown_face_image, face_locations, num_jitters=10)

        pil_image = Image.fromarray(unknown_face_image)
        draw = ImageDraw.Draw(pil_image)

        # Loop through each face found in the unknown image
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Draw a box around the face using the Pillow module
            draw.rectangle(((left, top), (right, bottom)),
                           outline=(0, 0, 255), width=4)

        # Remove the drawing library from memory as per the Pillow docs
        del draw

    return serve_pil_image(pil_image)


def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')


# Run Server
if __name__ == "__main__":
    from werkzeug.contrib.profiler import ProfilerMiddleware
    app.config['PROFILE'] = True # turn on/off profiler
    app.wsgi_app = ProfilerMiddleware(app.wsgi_app, profile_dir='./profile')
    app.run(debug=True)
