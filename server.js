const express = require('express');
const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs-node');
const cocoSsd = require('@tensorflow-models/coco-ssd');

const analytics = require('./analytics')

// HARDCODE uid
const uid = "RJ0pPZEpmqPdiwMNBsuErIKU8zI3";

// Init Tensorflow.js Obj Det model
let model;

cocoSsd.load().then((loadedModel) => {
  model = loadedModel;
  console.log('Coco SSD model loaded successfully');
}).catch((err) => {
  console.error('Error loading Coco SSD model:', err);
});

const uploadInterval = 10;

// Init firebase
const admin = require('firebase-admin');
const serviceAccount = require('./firebase_servicekey.json'); // Replace with the path to your service account key file

admin.initializeApp({
  credential: admin.credential.cert(serviceAccount),
});

const db = admin.firestore();

const activityRef = db.collection('users').doc(uid).collection('activity');

// Init Server
const app = express();
const uploadDir = path.join(__dirname, 'uploads');

// Create the uploads directory if it doesn't exist
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir);
}

past_pred = [];
var dogroll = new analytics.RollingAverage(uploadInterval);
var catroll = new analytics.RollingAverage(uploadInterval);
var personroll = new analytics.RollingAverage(uploadInterval);
var predictionsMade = 0;

// Currently ffmpeg streams 2 frames per second
// this script uploads to firebase every uploadInterval predicitions

app.post('/upload', (req, res) => {
  // console.time("timer read, predict, and delete file")

  const fileName = `frame_${Date.now()}.jpg`;
  const filePath = path.join(uploadDir, fileName);

  const fileStream = fs.createWriteStream(filePath);
  req.pipe(fileStream);

  fileStream.on('finish', () => {

    // console.log(`Image saved: ${fileName}`);

    // Object detection with TensorFlow.js
    new Promise((resolve, reject) => {
      const imageBuffer = fs.readFileSync(filePath);
      const imageTensor = tf.node.decodeImage(imageBuffer);
      resolve(imageTensor);
    }).then((imageTensor) => {
      return model.detect(imageTensor);
    }).then((predictions) => {
      // console.log('Object detection results:', predictions);

      // Analytics
      spred = analytics.scoreFilter(predictions)
      cspred = analytics.classFilter(spred)

      if (cspred.length > 0) {
        pred_center = analytics.centerpoint(cspred);
        // console.log('Predictions with center values:', pred_center);

        if (past_pred.length > 0) {
          distanceDifferences = analytics.velocity(past_pred, pred_center);
          
          for (i=0; i<distanceDifferences.length; i++) {
            distDiff = distanceDifferences[i];

            switch (distDiff.class) {
              case 'dog':
                dogroll.addValue(distDiff.difference)
                break;
              case 'cat':
                catroll.addValue(distDiff.difference)
                break;
              case 'person':
                personroll.addValue(distDiff.difference)
                break;
              case 'dining table': // only for testing
                dogroll.addValue(distDiff.difference)
                break;
              default:
                console.log('no class of interest found, only:',distDiff.class)
                break;
            }
            
          }

          console.log("Rolling avg. of dog activity:", dogroll.getAverage());
          console.log("Rolling avg. of cat activity:", catroll.getAverage());
          console.log("Rolling avg. of person activity:", personroll.getAverage());

        }

        past_pred = pred_center;

        predictionsMade++; 

        if (predictionsMade >= uploadInterval) {

          console.log("About to update firebase...")

          // upload to firebase
          const activityData = {
            dog: dogroll.getAverage(),
            cat: catroll.getAverage(),
            person: personroll.getAverage(),
            timestamp: admin.firestore.FieldValue.serverTimestamp(),
          };

          activityRef.add(activityData)
            .then((docRef) => {
              console.log('Activity data uploaded successfully:', docRef.id);
            })
            .catch((error) => {
              console.error('Error uploading activity data:', error);
          });

          predictionsMade = 0;
        }

      }

    }).catch((err) => {
      console.error('Error performing object detection:', err);
    });



    // After processing, delete the file
    fs.unlink(filePath, (err) => {
        if (err) {
        console.error('Error deleting file:', err);
        }
        console.log(`Image deleted: ${fileName}`);
    });

    // console.timeEnd("timer read, predict, and delete file")

    res.send('Image uploaded successfully!');

  });

  fileStream.on('error', (err) => {
    console.error('Error saving image:', err);
    res.status(500).send('Error saving image.');
  });
});

app.listen(3030, () => {
  console.log('Server listening on port 3030');
});
