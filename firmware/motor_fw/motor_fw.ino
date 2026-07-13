// Arduino Uno 모터+엔코더 펌웨어 (Jetson Option B)
// 시리얼 프로토콜 @115200:
//   Jetson -> Uno:  "M <left_pwm> <right_pwm>\n"   (-255..255)
//   Uno -> Jetson:  "E <left_ticks> <right_ticks>\n"  (20ms마다)

const uint8_t ENC_L_A = 2, ENC_L_B = 4;    // 왼쪽 엔코더 (A는 INT0)
const uint8_t ENC_R_A = 3, ENC_R_B = 5;    // 오른쪽 엔코더 (A는 INT1)
const uint8_t ENA = 9,  IN1 = 7,  IN2 = 8;    // 왼쪽 모터 (L298N) — 실배선 반영
const uint8_t ENB = 10, IN3 = 11, IN4 = 12;   // 오른쪽 모터 — 실배선 반영

volatile long ticksL = 0, ticksR = 0;

void isrL() { if (digitalRead(ENC_L_B)) ticksL++; else ticksL--; }
// 오른쪽 모터는 좌우 대칭 장착 → 전진 시 회전 방향이 반대이므로 부호 반전(전진=증가 유지)
void isrR() { if (digitalRead(ENC_R_B)) ticksR--; else ticksR++; }

void setMotor(uint8_t en, uint8_t inA, uint8_t inB, int pwm) {
  bool fwd = pwm >= 0;
  pwm = constrain(abs(pwm), 0, 255);
  digitalWrite(inA, fwd);
  digitalWrite(inB, !fwd);
  analogWrite(en, pwm);
}

long cmdL = 0, cmdR = 0;
unsigned long lastReport = 0, lastCmd = 0;

void setup() {
  Serial.begin(115200);
  pinMode(ENC_L_A, INPUT_PULLUP); pinMode(ENC_L_B, INPUT_PULLUP);
  pinMode(ENC_R_A, INPUT_PULLUP); pinMode(ENC_R_B, INPUT_PULLUP);
  pinMode(ENA, OUTPUT); pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(ENB, OUTPUT); pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);
  attachInterrupt(digitalPinToInterrupt(ENC_L_A), isrL, RISING);
  attachInterrupt(digitalPinToInterrupt(ENC_R_A), isrR, RISING);
}

void loop() {
  // ---- 명령 수신 ----
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    if (line.length() && line[0] == 'M') {
      int s1 = line.indexOf(' ');
      int s2 = line.indexOf(' ', s1 + 1);
      cmdL = line.substring(s1 + 1, s2).toInt();
      cmdR = line.substring(s2 + 1).toInt();
      lastCmd = millis();
    }
  }
  // ---- 안전장치: 300ms 명령 없으면 정지 ----
  if (millis() - lastCmd > 300) { cmdL = cmdR = 0; }

  setMotor(ENA, IN1, IN2, cmdL);
  setMotor(ENB, IN3, IN4, cmdR);

  // ---- 엔코더 회신 (20ms마다) ----
  if (millis() - lastReport >= 20) {
    lastReport = millis();
    long l, r;
    noInterrupts(); l = ticksL; r = ticksR; interrupts();
    Serial.print("E "); Serial.print(l); Serial.print(' '); Serial.println(r);
  }
}
