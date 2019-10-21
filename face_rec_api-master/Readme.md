Progress Report (10/10/2019)
The API is running and usable. There are three endpoints, including the ones for
training (base_url/train), recognization (base_url/recognize) and detection 
(base_url/detect). The default port of flask is 5000, the uploaded image file's
tag must be "image". See app.py in /cli for further detail.

API Usage (Python3 required):
1. pip install -r requirements.txt
2. python api.py

Testing Script
1. cd cli
2. python app.py

However, there are a couple of problems needed to be solved:
1. Solve redundant IO operation, loading/sending images
Currently, the server reads every image from the POST request and store it locally,
then re-open it for the PIL to use. Similarly, when the image is processed, the
server has to store it locally before sending it back to the client. These redundant
IO operations significantly affected the API's efficiency.

2. Solve persistent data storage (encoding faces)
For now, the encoded faces are stored in a python list buffer, which works for the
monolithic version, but is not going to work in micro-services, since different
services are not able to share memory, increasing the cost of storage. This will
possibly be solved by using a secondary storage tool, such as redis.

3. The requirements.txt file needs to be updated
Currently the requirements.txt file contains a lot of unnecessary libraries, which
must be removed ASAP for deployment and testing.

Progress Report (10/16/2019)