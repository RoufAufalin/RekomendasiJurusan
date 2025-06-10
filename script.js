let isModelLoaded = false
let model
let wordIndex = {}
let labelMap = {}

//parameter
let maxLen = 100 // Panjang maksimal input
let vocabSize = 10000 // Ukuran kosakata
const padding = "post"

async function loadModel() {
  try {
    model = await tf.loadLayersModel("tfjs_model/model.json")
    console.log("✅ Model loaded successfully")
  } catch (error) {
    console.error("❌ Failed to load model:", error)
    alert("Gagal memuat model. Periksa koneksi atau path ke file model.json.")
  }
}

async function loadLabelMap() {
  try {
    const res = await fetch("label_map.json")
    if (!res.ok) throw new Error("HTTP error " + res.status)
    labelMap = await res.json()
    console.log("✅ label_map loaded successfully")
  } catch (error) {
    console.error("❌ Failed to load label_map:", error)
    alert("Gagal memuat label_map.json")
  }
}

async function loadWordIndex() {
  try {
    const res = await fetch("word_index.json")
    if (!res.ok) throw new Error("HTTP error " + res.status)
    wordIndex = await res.json()
    console.log("✅ word_index loaded successfully")
  } catch (error) {
    console.error("❌ Failed to load word_index:", error)
    alert("Gagal memuat word_index.json. Pastikan file tersedia.")
  }
}

function preprocessText(text) {
  const words = text.toLowerCase().split(/\s+/)
  const sequence = words.map((word) => {
    return wordIndex[word] || wordIndex["<OOV>"] || 0
  })

  // Padding post (seperti pad_sequences(..., padding='post'))
  const maxLen = 100
  if (sequence.length < maxLen) {
    const padded = Array(maxLen).fill(0)
    for (let i = 0; i < sequence.length; i++) {
      padded[i] = sequence[i]
    }
    return padded
  } else {
    return sequence.slice(0, maxLen)
  }
}

async function predict() {
  const inputText = document.getElementById("inputText").value
  const tokens = preprocessText(inputText)
  const inputTensor = tf.tensor2d([tokens], [1, tokens.length])
  const prediction = model.predict(inputTensor)
  const predictedClass = prediction.argMax(-1).dataSync()[0]

  // 🔁 Mapping prediksi angka ke nama jurusan
  const predictedLabel = labelMap[predictedClass]

  document.getElementById(
    "result"
  ).innerText = `Prediksi Jurusan: (${predictedLabel})`
}

document.getElementById("predictBtn").addEventListener("click", predict)

async function init() {
  await loadWordIndex()
  await loadModel()
  await loadLabelMap()
}

init()
