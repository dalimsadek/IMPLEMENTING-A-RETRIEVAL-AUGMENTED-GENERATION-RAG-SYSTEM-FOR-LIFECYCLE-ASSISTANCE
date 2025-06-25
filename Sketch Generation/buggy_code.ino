#include <Arduino_TensorFlowLite.h>
#include <Arduino_LSM9DS1.h>

const char* gesture_model = "\x00\x00\x10\x03..."; // assume gesture_model.tflite is stored in program memory

const int kTensorArenaSize = 1024 * 2;
uint8_t tensorArena[kTensorArenaSize];

const int kAccelerometerAxis = 3;
const int kGyroscopeAxis = 3;

const float kAccelerometerScale = 0.061f;
const float kGyroscopeScale = 0.061f;

float accelerometerData[kAccelerometerAxis];
float gyroscopeData[kGyroscopeAxis];

tflite::Model* model = nullptr;
tflite::Tensor* inputTensor = nullptr;
tflite::Tensor* outputTensor = nullptr;

void setup() {
  Serial.begin(115200);
  LSM9DS1.begin();
  model = tflite::LoadModel(gesture_model);

  if (model == nullptr) {
    Serial.println("Failed to load model");
  }

  inputTensor = model->input(0);
  outputTensor = model->output(0);
}

void loop() {
  int16_t accelerometerRaw[kAccelerometerAxis];
  int16_t gyroscopeRaw[kGyroscopeAxis];

  LSM9DS1.readAcceleration(accelerometerRaw);
  LSM9DS1.readGyroscope(gyroscopeRaw);

  for (int i = 0; i < kAccelerometerAxis; i++) {
    accelerometerData[i] = accelerometerRaw[i] * kAccelerometerScale;
  }

  for (int i = 0; i < kGyroscopeAxis; i++) {
    gyroscopeData[i] = gyroscopeRaw[i] * kGyroscopeScale;
  }

  float inputData[kAccelerometerAxis + kGyroscopeAxis];

  for (int i = 0; i < kAccelerometerAxis; i++) {
    inputData[i] = accelerometerData[i];
  }

  for (int i = 0; i < kGyroscopeAxis; i++) {
    inputData[kAccelerometerAxis + i] = gyroscopeData[i];
  }

  inputTensor->data.f = inputData;

  TfLiteStatus invokeStatus = tflite::Invoke();

  if (invokeStatus != kTfLiteOk) {
    Serial.println("Invoke failed");
  }

  int8_t output = outputTensor->data.i8[0];
  Serial.println(output == 0 ? "punch" : "flex");

  delay(10);
}
