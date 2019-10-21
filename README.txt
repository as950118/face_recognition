1. 이미지를 반으로 줄여서 연산속도 증가
        # Read the file in bytes, reduce IO operation
        img_data = BytesIO(unknown_img.read())
        img = Image.open(img_data).convert('RGB')
        img = np.array(img)

        # Load the unknown picture and find all the faces and face encodings in the unknown image
        #unknown_face_image = face_recognition.load_image_file(img_data)
        unknown_face_image = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

2. face_location과 face_encoding에 각각 upsample과 jitters에 관한 인자 추가
        face_locations = face_recognition.face_locations(
            unknown_face_image, number_of_times_to_upsample=2)
        face_encodings = face_recognition.face_encodings(
            unknown_face_image, face_locations, num_jitters=10)

3. compare_face의 인자인 tolerance 조절
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding, tolerance=0.6)

4. 프로파일
pip install cprofilev
cprofilev -f {cprofile name}