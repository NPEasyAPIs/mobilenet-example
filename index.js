const express = require('express');
const helmet = require('helmet')
const fetch = require('node-fetch');

const tfnode = require('@tensorflow/tfjs-node');
const mobilenet = require('@tensorflow-models/mobilenet');

const app = express();
app.use(helmet());

async function load_classify_model() {
	const version = 2;
    const alpha = 0.5;
	return(await mobilenet.load({version, alpha}));
}

const classify_modelPromise = load_classify_model();

async function predict_url(img, res) {
  let response = await fetch(img);
  let imageBuf = await response.buffer();
  let decodedImage = tfnode.node.decodeImage(imageBuf, 3);
  let classify_model = await classify_modelPromise;
  let predictions = await classify_model.classify(decodedImage);
  res.status(200).send({data : predictions});
}

app.get('/', async(req, res) => {
  try {
	let img_path = req.query.img_url;
    await predict_url(img_path, res);
  }
  catch(err) {
	res.status(400).send({data : err.message});
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {});