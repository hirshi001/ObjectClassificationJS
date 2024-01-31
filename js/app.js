const video = document.querySelector("#videoElement");
const canvas = document.querySelector("#videoCanvas");

const b1 = document.querySelector("#b1");
const b2 = document.querySelector("#b2");
const b3 = document.querySelector("#b3");
const buttons = [b1, b2, b3]

const cb1 = document.querySelector("#cb1");
const cb2 = document.querySelector("#cb2");
const cb3 = document.querySelector("#cb3");
const buttonCountsTexts = [cb1, cb2, cb3]
const buttonCounts = [0, 0, 0]

const trainButton = document.querySelector("#trainButton");
const detectButton = document.querySelector("#detectButton");
const resetModelButton = document.querySelector("#resetModel");

const objectDetected = document.querySelector("#objectDetected");
const modelState = document.querySelector("#modelState");
const modelUntrained = "Untrained"
const modelTrained = "Trained"
const modelTraining = "Training..."

let canvasWidth = 400;
let canvasHeight = null;

let input_width = 100
let input_height = 100
let image_size = [input_height, input_width]
const CLASSES = 3
const dataset = []

let model = null


// Define a model for linear regression. The script tag makes `tf` available
// as a global variable.

function build_feature_extractor(inputs) {

  let x = tf.layers.conv2d({filters: 6, kernelSize: 3, activation: 'relu', inputShape: [input_height, input_width, 3]}).apply(inputs)
  x = tf.layers.averagePooling2d({ poolSize: [2, 2] }).apply(x)

  x = tf.layers.conv2d({filters: 16, kernelSize: 3, activation :'relu'}).apply(x)
  x = tf.layers.averagePooling2d({poolSize: [2, 2]}).apply(x)

  x = tf.layers.conv2d({filters: 32, kernelSize: 3, activation: 'relu'}).apply(x)
  x = tf.layers.averagePooling2d({poolSize: [2, 2]}).apply(x)

  return x
}

function build_model_adaptor(inputs) {
  let x = tf.layers.flatten().apply(inputs)
  x = tf.layers.dense({units: 32, activation: 'relu'}).apply(x)
  return x
}

function build_classifier_head(inputs) {
  return tf.layers.dense({units: CLASSES, activation: 'softmax', name: 'classifier_head'}).apply(inputs)
}

function build_data_augmentation(inputs){
  return inputs; // RandomContrast(factor = 0.2)(inputs)
}

function build_model(inputs) {

  let augmented = build_data_augmentation(inputs)

  let feature_extractor = build_feature_extractor(augmented)

  let model_adaptor = build_model_adaptor(feature_extractor)

  let classification_head = build_classifier_head(model_adaptor)

  let model = tf.model({inputs: inputs, outputs: classification_head})

  return model
}


function captureVideo(video) {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const canvasContext = canvas.getContext("2d");
  canvasContext.drawImage(video, 0, 0);
  return tf.browser.fromPixels(canvas);
}
function drawVideoOnCanvas(){
  const ctx = canvas.getContext('2d');

  let index = -1;
  // check if each button is clicked, and if it is, increment count
  for (let i = 0; i < buttons.length; i++) {
    if (buttons[i].pres) {
      buttonCounts[i]++;
      // update text of count button
      updateButtonCount(i)
      buttonCountsTexts[i].innerHTML = "Count: " + buttonCounts[i];
      index = i;
      break;
    }
  }

  let resized = null
  if(index !== -1) {
    // get image data from video without using canvas, and save it to a dataset to be trained later
    const frame = captureVideo(video)
    resized = frame.resizeBilinear([input_height, input_width])
    dataset.push([resized, index])
  }
  ctx.drawImage(video, 0, 0, canvasWidth, canvasHeight);

  setTimeout(drawVideoOnCanvas , 50);
}

function updateButtonCount(index) {
  buttonCountsTexts[index].innerHTML = "Count: " + buttonCounts[index];
}

function updateButtonCounts() {
  for (let i = 0; i < buttons.length; i++) {
    updateButtonCount(i)
  }
}

function drawRect(ctx, x, y, w, h){
  ctx.rect(x,y,w,h);
  ctx.lineWidth = "6";
  ctx.strokeStyle = "red";
  ctx.stroke();
}

function prepareVideoStream() {
  if (navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(function (stream) {
        video.srcObject = stream;
        console.log("video stream ready")
      })
      .catch(function (error) {
        console.log("Something went wrong!");
        console.log(error)
      });
  }

  video.onplay = function() {
    canvasHeight = video.videoHeight * canvasWidth / video.videoWidth;
    canvas.width = canvasWidth
    canvas.height = canvasHeight
    setTimeout(drawVideoOnCanvas , 300);
  };

}

function processImage(image) {
  return image.toFloat().div(tf.scalar(255));
}


function* dataGenerator() {
  dataset.sort(() => Math.random() - 0.5)
  for (let i = 0; i < dataset.length; i++) {

    const tensor = processImage(dataset[i][0].reshape([input_height, input_width, 3]));
    const label = tf.oneHot(dataset[i][1], CLASSES); // Assuming dataset[i][1] is the label

    yield {xs: tensor, ys: label, done: i === dataset.length - 1}
  }
}

async function trainModel() {
  modelState.innerHTML = modelTraining
  // create dataset
  const dataset = tf.data.generator(dataGenerator).batch(10)
  // console.log(dataset.take(1))

  // train model
  await model.fitDataset(dataset, {
    epochs: 10,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        if (logs.acc !== undefined) {
          console.log(`Epoch ${epoch + 1}/${10}, Loss: ${logs.loss}, Accuracy: ${logs.acc}`);
        } else {
          console.log(`Epoch ${epoch + 1}/${10}, Loss: ${logs.loss}`);
        }
      }
    }
  });
  modelState.innerHTML = modelTrained

  /*
  // Split tensors into features (X) and labels (Y)
  const X = tensors.map(([tensor]) => tensor);
  const Y = tensors.map(([, label]) => label);

  // Convert labels to one-hot encoding (assuming labels are integer values)
  const oneHotLabels = tf.oneHot(tf.tensor1d(Y, 'int32'), );

  // Train the model

  model.fit(X, oneHotLabels, {
    epochs: 10,
    shuffle: true,
    batchSize: 16,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(`Epoch ${epoch + 1}/${10}, Loss: ${logs.loss.toFixed(4)}, Accuracy: ${logs.acc.toFixed(4)}`);
      }
    }
  });
  */

}

async function detect() {
  const predictions = model.predict(tf.expandDims(processImage(captureVideo(video).resizeBilinear([input_height, input_width])), 0));
  const object = predictions.argMax(1).dataSync()[0] + 1;
  objectDetected.innerHTML = "Object: " + object
}

function setButtonListeners() {

  // set image save buttons
  for (let i = 0; i < buttons.length; i++) {
    buttons[i].addEventListener('mousedown', (function(index) {
      return function() {
        buttons[index].pres = true;
      };
    })(i));

    buttons[i].addEventListener('mouseup', (function(index) {
      return function() {
        buttons[index].pres = false;
      };
    })(i));
  }

  // set train button
  trainButton.addEventListener('click', () => {
    trainModel()
  })

  // set detect button
  detectButton.addEventListener('click', () => {
    console.log("detecting")
    detect()
  })

  // set reset model button
  resetModelButton.addEventListener('click', () => {
      resetAll()
  })

}

function resetAll() {
  setModel()
  for(let i = 0; i < buttonCounts.length; i++){
    buttonCounts[i] = 0
  }
  updateButtonCounts()
}

function setModel() {
  model = build_model(tf.input({shape: [input_height, input_width, 3]}))
  model.compile({optimizer: tf.train.adam(0.001), loss: 'categoricalCrossentropy', metrics: ['accuracy']})
  modelState.innerHTML = modelUntrained
  objectDetected.innerHTML = "Object: None"
}
function onLoad() {
  prepareVideoStream()
  setModel()
  setButtonListeners()

}


