function classFilter(predictions) {

    // Output format:
    // tensorflow predicition array

    var passedPredictions = [];
    acceptable_classes = ['dog','cat','person','dining table'] // dining table for testing since ffmpeg test footage gives this predicition lul

    for (var i = 0; i < predictions.length; i++) {
      if (acceptable_classes.includes(predictions[i].class)) {
        passedPredictions.push(predictions[i]);
      }
    }
    
    return passedPredictions
}

function scoreFilter(predictions) {

    // Output format:
    // tensorflow predicition array

    var passedPredictions = [];

    for (var i = 0; i < predictions.length; i++) {
      if (predictions[i].score > 0.5) {
        passedPredictions.push(predictions[i]);
      }
    }
    
    return passedPredictions

}

function centerpoint(predictions) {

    // Data format:
    // bbox: [x, y, width, height]
    //
    //     (x, y)       (x + width, y)
    //     +----------------------+
    //     |                      |
    //     |                      |
    //     |                      |
    //     +----------------------+
    // (x, y + height)     (x + width, y + height)

    // Output format:
    // tensorflow predicition array with 'center' field

    var centerPointPredictions = [];

    // calculate the center point of a bounding box
    for (var i = 0; i < predictions.length; i++) {
        bbox = predictions[i].bbox
        xc = bbox[0] + ( bbox[2] / 2 );
        yc = bbox[1] + ( bbox[3] / 2 );


        centerPointPredictions.push({
            ...predictions[i], // Copy the existing prediction object
            center: [xc, yc] // Add the "center" field with the center coordinates
        });
    }
    return centerPointPredictions
}

function velocity(pastPositions, currentPositions) {

    // Calculate the velocity of a bounding box between two predictions (change in pixels / change in prediction)

    // Output format:
    // [ { class: 'dining table', difference: 4.072856903076172 } ]

    // Calculate difference in distance for matched classes
    const distanceDifferences = [];

    // Iterate over the past positions
    pastPositions.forEach((pastPosition) => {
    const { class: pastClass, center: pastCenter } = pastPosition;

    // Find the matching current position by class
    const currentPosition = currentPositions.find((currentPosition) => currentPosition.class === pastClass);

    // If a matching position is found
    if (currentPosition) {
        const { center: currentCenter } = currentPosition;

        // Calculate the difference in distance for the center values
        const distanceDifference = Math.abs(currentCenter[0] - pastCenter[0]) + Math.abs(currentCenter[1] - pastCenter[1]);

        // Add the difference to the array
        distanceDifferences.push({
        class: pastClass,
        difference: distanceDifference
        });
    }
    });

    return distanceDifferences
}

function RollingAverage(windowSize) {
    this.windowSize = windowSize;
    this.values = [];
    this.sum = 0;

    // // Example usage:
    // var average = new RollingAverage(5);

    // average.addValue(10);
    // average.addValue(15);
    // average.addValue(20);
    // average.addValue(25);
    // average.addValue(30);

    // console.log(average.getAverage()); // Output: 20

    // average.addValue(35);
    // console.log(average.getAverage()); // Output: 25

}

RollingAverage.prototype.addValue = function(value) {
    this.values.push(value);
    this.sum += value;

    // Remove the oldest value if the window size is exceeded
    if (this.values.length > this.windowSize) {
        var removedValue = this.values.shift();
        this.sum -= removedValue;
    }
};

RollingAverage.prototype.getAverage = function() {
    return this.sum / this.values.length;
};

module.exports = {
    classFilter,
    scoreFilter,
    centerpoint,
    velocity,
    RollingAverage
}