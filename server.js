const express = require('express');
const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs');
const cocoSsd = require('@tensorflow-models/coco-ssd');

// Init Tensorflow.js Obj Det model
let model;

cocoSsd.load().then((loadedModel) => {
  model = loadedModel;
  console.log('Coco SSD model loaded successfully');
}).catch((err) => {
  console.error('Error loading Coco SSD model:', err);
});

// Init Server
const app = express();
const uploadDir = path.join(__dirname, 'uploads');

// Create the uploads directory if it doesn't exist
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir);
}

app.post('/upload', (req, res) => {
  const fileName = `frame_${Date.now()}.jpg`;
  const filePath = path.join(uploadDir, fileName);

  const fileStream = fs.createWriteStream(filePath);
  req.pipe(fileStream);

  fileStream.on('finish', () => {

    console.log(`Image saved: ${fileName}`);

    // Object detection with Tensorflow.js
    tf.node.decodeImage(fs.readFileSync(filePath)).then((imageTensor) => {
        return model.detect(imageTensor);
    }).then((predictions) => {
        console.log('Object detection results:', predictions);
        // Process the predictions as needed
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
