// --- 키보드(시리얼)로 좌/우 PWM을 개별 제어하는 테스트 스케치 ---
// 사용법: 시리얼 모니터를 115200으로 열고, 한 줄 입력 후 Enter
//   "100 130"   → 왼쪽 PWM=100, 오른쪽 PWM=130 (둘 다 전진)
//   "-100 130"  → 왼쪽 후진 100, 오른쪽 전진 130 (음수 = 후진)
//   "150"       → 양쪽 모두 150
//   w=전진  s=후진  a=좌회전  d=우회전  x=정지  (마지막 속도 크기 사용)
//   + = 속도 +10   - = 속도 -10   ? = 도움말
// ⚠ 바퀴를 바닥에서 띄우고 테스트하세요.

// --- 핀 설정 ---
// 왼쪽 모터 (Motor A)
const int ENA = 9;
const int IN1 = 7;
const int IN2 = 8;
const int ENCA_A = 2; // 하드웨어 인터럽트 0
const int ENCA_B = 4;

// 오른쪽 모터 (Motor B)
const int ENB = 10;
const int IN3 = 11;
const int IN4 = 12;
const int ENCB_A = 3; // 하드웨어 인터럽트 1
const int ENCB_B = 5;

// --- 상태 변수 ---
volatile long leftCount = 0;
volatile long rightCount = 0;

int pwmL = 0, pwmR = 0;      // 부호 있는 PWM (-255~255, 음수=후진)
int speed = 150;             // w/s/a/d 에 쓰이는 속도 크기
char lineBuf[24];
uint8_t lineLen = 0;
unsigned long lastReport = 0;

void setup() {
  Serial.begin(115200);

  pinMode(ENA, OUTPUT); pinMode(ENB, OUTPUT);
  pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT); pinMode(IN4, OUTPUT);

  // 엔코더: 안정적인 읽기를 위해 내부 풀업 사용
  pinMode(ENCA_A, INPUT_PULLUP); pinMode(ENCA_B, INPUT_PULLUP);
  pinMode(ENCB_A, INPUT_PULLUP); pinMode(ENCB_B, INPUT_PULLUP);

  attachInterrupt(digitalPinToInterrupt(ENCA_A), readLeftEncoder, RISING);
  attachInterrupt(digitalPinToInterrupt(ENCB_A), readRightEncoder, RISING);

  printHelp();
}

void loop() {
  // --- 한 줄 단위 입력 수집 ---
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (lineLen > 0) {
        lineBuf[lineLen] = '\0';
        processLine(lineBuf);
        lineLen = 0;
      }
    } else if (lineLen < sizeof(lineBuf) - 1) {
      lineBuf[lineLen++] = c;
    }
  }

  // --- 엔코더 주기 출력 (300ms) ---
  if (millis() - lastReport >= 300) {
    lastReport = millis();
    Serial.print(F("Left Ticks: "));  Serial.print(leftCount);
    Serial.print(F("  |  Right Ticks: ")); Serial.print(rightCount);
    Serial.print(F("  |  PWM L=")); Serial.print(pwmL);
    Serial.print(F(" R=")); Serial.println(pwmR);
  }
}

// --- 한 줄 명령 처리 ---
void processLine(char *s) {
  while (*s == ' ' || *s == '\t') s++;   // 앞 공백 제거
  if (!*s) return;
  char c0 = *s;

  if      (c0 == 'w' || c0 == 'W') { pwmL =  speed; pwmR =  speed; }   // 전진
  else if (c0 == 's' || c0 == 'S') { pwmL = -speed; pwmR = -speed; }   // 후진
  else if (c0 == 'a' || c0 == 'A') { pwmL = -speed; pwmR =  speed; }   // 제자리 좌회전
  else if (c0 == 'd' || c0 == 'D') { pwmL =  speed; pwmR = -speed; }   // 제자리 우회전
  else if (c0 == 'x' || c0 == 'X') { pwmL = 0; pwmR = 0; }             // 정지
  else if (c0 == '?')              { printHelp(); return; }
  else if (c0 == '+' && s[1] == '\0') {   // 단독 '+' : 속도 크기 +10
    speed = min(speed + 10, 255);
    Serial.print(F(">> speed=")); Serial.println(speed);
    return;
  }
  else if (c0 == '-' && s[1] == '\0') {   // 단독 '-' : 속도 크기 -10
    speed = max(speed - 10, 0);
    Serial.print(F(">> speed=")); Serial.println(speed);
    return;
  }
  else if (c0 == '-' || c0 == '+' || (c0 >= '0' && c0 <= '9')) {
    // 숫자 입력: "L R" 두 개면 개별 설정, 하나면 양쪽 동일
    char *end;
    long a = strtol(s, &end, 10);
    if (end == s) return;                 // 숫자 아님
    char *p = end;
    while (*p == ' ' || *p == ',' || *p == '\t') p++;
    if (*p) {                             // 두 번째 숫자 존재
      char *end2;
      long b = strtol(p, &end2, 10);
      if (end2 == p) return;
      pwmL = constrain(a, -255, 255);
      pwmR = constrain(b, -255, 255);
    } else {                              // 숫자 하나 → 양쪽 동일
      pwmL = pwmR = constrain(a, -255, 255);
    }
    int m = max(abs(pwmL), abs(pwmR));    // w/s/a/d 용 속도 크기 갱신
    if (m > 0) speed = m;
  }
  else return;                            // 그 외 무시

  applyMotors();
  Serial.print(F(">> PWM L=")); Serial.print(pwmL);
  Serial.print(F(" R=")); Serial.println(pwmR);
}

// --- 현재 PWM을 모터에 반영 ---
void applyMotors() {
  driveMotor(IN1, IN2, ENA, pwmL);
  driveMotor(IN3, IN4, ENB, pwmR);
}

// pwm: -255~255 (음수=후진, 0=정지)
void driveMotor(int inA, int inB, int en, int pwm) {
  if (pwm > 0)      { digitalWrite(inA, HIGH); digitalWrite(inB, LOW);  }
  else if (pwm < 0) { digitalWrite(inA, LOW);  digitalWrite(inB, HIGH); }
  else              { digitalWrite(inA, LOW);  digitalWrite(inB, LOW);  }
  analogWrite(en, min(abs(pwm), 255));
}

void printHelp() {
  Serial.println(F("=== 좌/우 PWM 개별 제어 (115200) ==="));
  Serial.println(F("\"100 130\"+Enter  → 왼쪽 PWM=100, 오른쪽 PWM=130"));
  Serial.println(F("\"-100 130\"       → 왼쪽 후진, 오른쪽 전진 (음수=후진)"));
  Serial.println(F("\"150\"            → 양쪽 모두 150"));
  Serial.println(F("w=전진 s=후진 a=좌회전 d=우회전 x=정지"));
  Serial.println(F("+ =속도+10  - =속도-10  ? =도움말"));
}

// --- 엔코더 ISR ---
// 왼쪽 엔코더
void readLeftEncoder() {
  if (digitalRead(ENCA_B)) leftCount++; else leftCount--;
}
// 오른쪽 엔코더 (좌우 대칭 장착 보정: 전진 시 +가 되도록 부호 반전)
void readRightEncoder() {
  if (digitalRead(ENCB_B)) rightCount--; else rightCount++;
}
