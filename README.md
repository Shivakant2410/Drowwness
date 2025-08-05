# Drowsiness Detection System

A real-time drowsiness detection system using computer vision and deep learning techniques to monitor driver alertness and prevent accidents caused by fatigue.

## Features

- **Real-time Face Detection**: Uses MediaPipe for accurate face landmark detection
- **Eye Aspect Ratio (EAR) Calculation**: Monitors eye closure patterns to detect drowsiness
- **Age Prediction**: Estimates the age of detected individuals using a deep learning model
- **Multi-face Support**: Can detect and monitor up to 5 faces simultaneously
- **Visual Feedback**: Color-coded bounding boxes and labels for drowsy/awake states
- **GUI Application**: User-friendly interface for easy interaction

## Technology Stack

- **Python 3.x**
- **OpenCV**: Computer vision operations
- **PyTorch**: Deep learning framework for age prediction model
- **MediaPipe**: Face mesh detection and landmark extraction
- **Torchvision**: Image preprocessing and transformations
- **NumPy**: Numerical computations

## Project Structure

```
├── age_model.py          # Age prediction model architecture
├── drowsiness_util.py    # Core utility functions for drowsiness detection
├── gui_app.py           # GUI application interface
├── test3.py             # Testing and demonstration script
├── age_model.pt         # Pre-trained age prediction model (not included in repo)
├── .gitignore           # Git ignore file
└── README.md            # Project documentation
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Shivakant2410/Drowwness.git
   cd Drowwness
   ```

2. **Install required dependencies**:
   ```bash
   pip install opencv-python
   pip install torch torchvision
   pip install mediapipe
   pip install numpy
   pip install tkinter  # Usually comes with Python
   ```

3. **Download or train the age prediction model**:
   - The `age_model.pt` file is required but not included in the repository due to size constraints
   - You'll need to either train your own model or obtain a pre-trained model
   - Place the model file in the project root directory

## Usage

### Running the GUI Application
```bash
python gui_app.py
```

### Running the Test Script
```bash
python test3.py
```

### Using the Core Functions
```python
from drowsiness_util import detect_faces_and_drowsiness, load_model, get_transform
from age_model import AgeModel  # Your model class

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model(AgeModel, 'age_model.pt', device)
transform = get_transform()

# Process a frame
frame, sleepy_count, drowsy_ages = detect_faces_and_drowsiness(frame, model, transform, device)
```

## How It Works

### Drowsiness Detection Algorithm

1. **Face Detection**: Uses MediaPipe Face Mesh to detect facial landmarks
2. **Eye Aspect Ratio (EAR) Calculation**: 
   - Calculates the ratio of eye height to eye width
   - EAR decreases significantly when eyes are closed
   - Threshold of 0.22 is used to determine drowsiness
3. **Age Prediction**: Deep learning model predicts age from face crops
4. **Alert System**: Visual indicators show drowsy/awake status

### EAR Formula
```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```
Where p1, p2, p3, p4, p5, p6 are facial landmarks around the eye.

## Key Functions

### `drowsiness_util.py`
- `load_model()`: Loads and initializes the age prediction model
- `get_transform()`: Returns image preprocessing transformations
- `compute_ear()`: Calculates Eye Aspect Ratio from landmarks
- `detect_faces_and_drowsiness()`: Main detection function

### `age_model.py`
- Contains the neural network architecture for age prediction

### `gui_app.py`
- Provides a graphical user interface for the application

## Model Requirements

The system requires a pre-trained PyTorch model (`age_model.pt`) for age prediction. The model should:
- Accept 100x100 RGB images as input
- Output a single age value
- Be compatible with the `AgeModel` class architecture

## Configuration

You can adjust the drowsiness threshold by modifying the EAR threshold in `drowsiness_util.py`:
```python
if ear < 0.22:  # Adjust this value as needed
    # Person is considered drowsy
```

## Performance Considerations

- **GPU Support**: The system automatically uses CUDA if available
- **Real-time Processing**: Optimized for real-time video processing
- **Multi-face Detection**: Can handle up to 5 faces simultaneously
- **Memory Efficient**: Uses efficient tensor operations

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).

## Future Enhancements

- [ ] Add audio alerts for drowsiness detection
- [ ] Implement drowsiness severity levels
- [ ] Add data logging and analytics
- [ ] Mobile app integration
- [ ] Real-time dashboard for monitoring
- [ ] Integration with vehicle systems

## Troubleshooting

### Common Issues

1. **Model file not found**: Ensure `age_model.pt` is in the project directory
2. **Camera not detected**: Check camera permissions and connections
3. **Poor detection accuracy**: Ensure good lighting conditions
4. **Performance issues**: Consider using GPU acceleration

### Dependencies Issues
If you encounter dependency conflicts, create a virtual environment:
```bash
python -m venv drowsiness_env
source drowsiness_env/bin/activate  # On Windows: drowsiness_env\Scripts\activate
pip install -r requirements.txt
```

## Contact

- **Author**: Shivakant
- **GitHub**: [@Shivakant2410](https://github.com/Shivakant2410)
- **Repository**: [Drowwness](https://github.com/Shivakant2410/Drowwness)

## Acknowledgments

- MediaPipe team for the face detection framework
- PyTorch community for the deep learning framework
- OpenCV contributors for computer vision tools
