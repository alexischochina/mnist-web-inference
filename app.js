// Variables globales
let isDrawing = false;
let context = null;
let model = null;

// Initialisation
window.addEventListener('load', () => {
    // Initialiser le canvas
    const canvas = document.getElementById('canvas');
    context = canvas.getContext('2d');
    context.lineWidth = 15;
    context.lineCap = 'round';
    context.strokeStyle = 'black';
    context.fillStyle = 'white';
    context.fillRect(0, 0, canvas.width, canvas.height);

    // Événements de dessin
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);

    // Événements tactiles
    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent('mousedown', {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        canvas.dispatchEvent(mouseEvent);
    });

    canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent('mousemove', {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        canvas.dispatchEvent(mouseEvent);
    });

    canvas.addEventListener('touchend', (e) => {
        e.preventDefault();
        const mouseEvent = new MouseEvent('mouseup', {});
        canvas.dispatchEvent(mouseEvent);
    });

    // Boutons
    document.getElementById('predict').addEventListener('click', predict);
    document.getElementById('clear').addEventListener('click', clearCanvas);

    // Charger le modèle ONNX
    loadModel();
});

// Fonctions de dessin
function startDrawing(e) {
    isDrawing = true;
    draw(e);
}

function draw(e) {
    if (!isDrawing) return;
    
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    context.beginPath();
    context.moveTo(lastX || x, lastY || y);
    context.lineTo(x, y);
    context.stroke();
    
    [lastX, lastY] = [x, y];
}

function stopDrawing() {
    isDrawing = false;
    [lastX, lastY] = [null, null];
}

function clearCanvas() {
    context.fillStyle = 'white';
    context.fillRect(0, 0, canvas.width, canvas.height);
    document.getElementById('result').textContent = '';
}

// Chargement du modèle ONNX
async function loadModel() {
    try {
        model = await ort.InferenceSession.create('models/mnist.onnx');
        console.log('Modèle chargé avec succès');
    } catch (e) {
        console.error('Erreur lors du chargement du modèle:', e);
    }
}

// Prétraitement de l'image
function preprocessImage() {
    // Redimensionner à 28x28
    const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempContext = tempCanvas.getContext('2d');
    tempContext.drawImage(canvas, 0, 0, 28, 28);
    
    // Convertir en tableau de pixels normalisés
    const imageDataResized = tempContext.getImageData(0, 0, 28, 28);
    const input = new Float32Array(1 * 1 * 28 * 28);
    
    for (let i = 0; i < imageDataResized.data.length; i += 4) {
        // Convertir en niveaux de gris et normaliser
        const pixel = imageDataResized.data[i];
        input[i/4] = (255 - pixel) / 255.0;
    }
    
    return input;
}

// Prédiction
async function predict() {
    if (!model) {
        alert('Le modèle n\'est pas encore chargé');
        return;
    }

    // Prétraiter l'image
    const input = preprocessImage();
    
    try {
        // Préparer le tenseur d'entrée
        const tensor = new ort.Tensor('float32', input, [1, 1, 28, 28]);
        
        // Faire la prédiction
        const results = await model.run({ 'input': tensor });
        const output = results['output'].data;
        
        // Trouver la classe prédite
        const predictedClass = output.indexOf(Math.max(...output));
        
        // Afficher le résultat
        document.getElementById('result').textContent = `Prédiction : ${predictedClass}`;
    } catch (e) {
        console.error('Erreur lors de la prédiction:', e);
        alert('Erreur lors de la prédiction');
    }
} 