# Global Health Data Compliance Engineering Policy

**Scope:** This document defines mandatory engineering policies and technical requirements for building and operating a technology product intended for use by licensed psychotherapists and other healthcare professionals in the **United States, European Union, United Kingdom, Canada, and Australia**.

**Covered Regulations:**

* HIPAA (US)
* GDPR (EU) / UK GDPR
* PIPEDA (Canada)
* Australian Privacy Act (APPs)
* SOC 2 Type II
* ISO 27001

This policy is binding for all engineers, contractors, and technical leadership.

---

## 1. Core Principles (Non-Negotiable)

1. **Privacy by Design and Default**

   * Security and privacy controls must be enabled by default.
   * No feature may weaken confidentiality without explicit user action.

2. **Least Privilege Everywhere**

   * Access is granted only to the minimum data and systems required.

3. **Data Minimization**

   * Collect, process, and retain the minimum data necessary for the clinical function.

4. **No Secondary Use of Health Data**

   * Client data **must not** be used for analytics, advertising, or AI training unless explicitly enabled by the customer with documented consent.

---

## 2. Data Classification Policy

All data must be classified at ingestion.

### 2.1 Data Classes

* **PHI / Health Data (Highest Sensitivity)**

  * Therapy notes, session metadata, diagnoses, assessments, recordings
* **Personal Data**

  * Names, emails, IP addresses, billing info
* **Operational Metadata**

  * Logs, metrics, anonymized usage data

### 2.2 Handling Rules

* PHI must **never** be logged in plaintext.
* PHI must be encrypted at rest and in transit.
* PHI must be logically isolated from non-PHI systems.

---

## 3. Identity, Authentication, and Access Control

### 3.1 User Authentication

* MFA required for all privileged and clinical accounts.
* Passwords must meet NIST SP 800-63 guidelines.
* Credentials must be stored using modern adaptive hashing (Argon2id or bcrypt).

### 3.2 Authorization

* Role-Based Access Control (RBAC) is mandatory.
* Roles must be granular (e.g., Therapist, Supervisor, Admin, Support).
* Support access to PHI requires explicit, time-bound approval.

### 3.3 Session Management

* Automatic session expiration after inactivity.
* Secure, HttpOnly, SameSite cookies.

---

## 4. Encryption Standards

### 4.1 Data in Transit

* TLS 1.2 minimum; TLS 1.3 preferred.
* Certificate pinning for mobile clients where feasible.

### 4.2 Data at Rest

* AES-256 or equivalent encryption for all databases and backups.
* Encryption keys managed via approved KMS/HSM services.
* No hard-coded secrets or keys in source control.

### 4.3 Key Management

* Key rotation at least annually.
* Immediate rotation on suspected compromise.

---

## 5. Audit Logging and Monitoring

### 5.1 Audit Logs (Required)

* Access to PHI (read/write/delete)
* Authentication events
* Permission changes
* Data exports

### 5.2 Log Properties

* Immutable or tamper-evident
* Time-synchronized (UTC)
* Retained for **minimum 6 years**

### 5.3 Monitoring

* Automated alerts for anomalous access patterns.
* Failed login and privilege escalation detection.

---

## 6. Data Subject Rights Support

The system **must support** the following operations:

### 6.1 Access & Portability

* Export of client data in a structured, machine-readable format.

### 6.2 Rectification

* Ability to correct inaccurate data without destroying audit history.

### 6.3 Erasure & Restriction

* Configurable deletion workflows respecting jurisdictional retention laws.
* Soft-delete with cryptographic erasure preferred.

---

## 7. Data Residency and Cross-Border Transfers

* Regional data storage must be supported (US, EU/UK, Canada, Australia).
* Cross-region transfers must be explicitly controlled and logged.
* Standard contractual safeguards must be assumed for EU/UK data.

---

## 8. Secure Development Practices

### 8.1 SDLC Requirements

* Threat modeling required for all new features touching PHI.
* Security review required before production release.

### 8.2 Code Standards

* Input validation and output encoding everywhere.
* Use of approved cryptographic libraries only.
* No dynamic code execution on untrusted input.

### 8.3 Dependencies

* Continuous dependency vulnerability scanning.
* High/critical CVEs must block release.

---

## 9. Incident Response and Breach Handling

### 9.1 Detection

* Breach detection must be automated where possible.

### 9.2 Response

* Incidents must be triaged within 24 hours.
* Containment and mitigation steps documented.

### 9.3 Notification Readiness

* Ability to notify customers within **72 hours**.
* Jurisdiction-specific reporting workflows maintained.

---

## 10. AI and Automated Processing Restrictions

* AI features must be **opt-in** for healthcare data.
* No model training on client data by default.
* Automated decision-making affecting clients must be explainable and reviewable.

---

## 11. Vendor and Subprocessor Controls

* All subprocessors must meet equivalent security standards.
* Written agreements required before PHI access.
* Annual security review of critical vendors.

---

## 12. Compliance Verification

* Annual security risk assessment required.
* External audit readiness (HIPAA, GDPR, SOC 2).
* Engineers must participate in compliance training annually.

---

## 13. Enforcement

Failure to follow this policy may result in:

* Removal of system access
* Disciplinary action
* Contract termination

This policy exists to protect clients, clinicians, and the organization. Compliance is an engineering responsibility, not a legal afterthought.
