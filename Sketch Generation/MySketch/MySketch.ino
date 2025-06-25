void setup() {
  Serial.begin(9600)
  pinMode(13, OUTPUT);  // missing comma between arguments
}

void loop() {
  digitalWrite(13, "HIGH");
  delay(1000);
}
