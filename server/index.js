const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const { pipeline } = require('transformers');

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

// Initialize the model
let model;
async function loadModel() {
  model = await pipeline('text-classification', {
    model: './models/bert_classifier_model',
  });
}

loadModel();

app.post('/api/analyze', async (req, res) => {
  try {
    const { text } = req.body;
    
    if (!text) {
      return res.status(400).json({ error: 'Text is required' });
    }

    const result = await model(text);
    
    res.json({
      prediction: result[0].label,
      confidence: result[0].score
    });
  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ error: 'Analysis failed' });
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
}); 