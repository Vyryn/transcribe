# Offline-First Global Health Data Compliance – Engineering Checklist

**Purpose:**
This checklist defines the *minimum and sufficient* engineering requirements for an **offline-only** product intended for use by licensed psychotherapists in the **US, EU, UK, Canada, and Australia**.

**Assumption:**

* The product **does not communicate with the internet** during normal operation.
* No cloud services, telemetry, remote analytics, or background network calls are permitted.

If a requirement is not met, the product is **not compliant for clinical use**.

---

## 1. Network Isolation (FOUNDATIONAL)

☐ The application functions fully with **no network connectivity**
☐ No outbound network calls at runtime (including analytics, updates, crash reporting)
☐ No hard-coded URLs, endpoints, or remote dependencies
☐ Network access is not required for authentication or licensing

> Any network capability immediately expands regulatory scope and risk.

---

## 2. Data Classification & Storage

☐ All stored data is assumed to be **PHI / Sensitive Health Data**
☐ Data is stored **only on the local device**
☐ No plaintext storage of therapy notes or session data
☐ Temporary files, caches, and backups follow the same protection rules

---

## 3. Encryption (MANDATORY)

### Data at Rest

☐ All persistent storage encrypted (AES-256 or equivalent)
☐ Full-disk encryption supported or enforced where possible
☐ Encryption applies to:

* Databases
* Files
* Backups
* Exports

### Key Management

☐ Encryption keys never hard-coded
☐ Keys derived from user credentials or OS secure storage
☐ Key material wiped on account deletion

---

## 4. Authentication & Local Access Control

☐ User authentication required to access the application
☐ Strong password or passphrase enforced
☐ Biometric unlock allowed **only as a convenience layer**, not sole protection
☐ Automatic lock after configurable inactivity

☐ Separate local profiles supported (no shared accounts)

---

## 5. Authorization (If Multi-User)

☐ Role separation implemented (e.g., Therapist vs Admin)
☐ Access to client records restricted by role
☐ No background or silent privilege escalation

---

## 6. Audit Logging (LOCAL)

☐ Local audit log records:

* Record access
* Record modification
* Deletion events
* Export events

☐ Logs are:

* Tamper-evident
* Time-stamped (local time + timezone)
* Retained for at least **6 years** unless user deletes entire profile

☐ Logs never include plaintext PHI

---

## 7. Data Integrity & Change History

☐ Accidental deletion is preventable or reversible
☐ Record modifications preserve historical versions or change metadata
☐ Integrity checks detect corruption or unauthorized modification

---

## 8. Data Export & Portability

☐ User can export all client data on demand
☐ Export format is structured and readable (e.g., JSON, PDF + metadata)
☐ Exported data is encrypted by default
☐ Clear warning displayed before exporting sensitive data

---

## 9. Data Deletion & Retention

☐ User can delete individual client records
☐ User can delete entire local profile
☐ Deletion performs secure wipe or cryptographic erasure
☐ Retention behavior is documented for clinicians

> Legal retention responsibility rests with the clinician, not the software.

---

## 10. Secure Development Practices

☐ No debug backdoors or hidden access paths
☐ No logging of sensitive data during development or production
☐ Secure coding practices enforced (input validation, memory safety)
☐ Third-party libraries reviewed and minimized

---

## 11. AI / Automated Features (If Present)

☐ All AI features operate **entirely on-device**
☐ No model training on user data
☐ AI outputs are advisory only (no autonomous clinical decisions)
☐ Clear disclosure of AI limitations

---

## 12. Platform Security Integration

☐ Uses OS secure storage / keychain where available
☐ Respects OS-level encryption and sandboxing
☐ Data inaccessible to other applications on the device

---

## 13. Compliance Alignment Summary

If all items above are satisfied, the product is **aligned with**:

* HIPAA (Security Rule – local safeguards)
* GDPR / UK GDPR (privacy by design, data minimization)
* PIPEDA (appropriate safeguards)
* Australian Privacy Act (sensitive information protection)

No cloud usage means:

* No cross-border transfer risk
* No breach notification infrastructure requirement
* Significantly reduced compliance surface

---

## 14. Final Engineering Sign-Off

☐ No network access verified
☐ Encryption verified
☐ Access control verified
☐ Data export & deletion verified
☐ Checklist reviewed before release

**Offline-first is not a shortcut — it is a security posture.**
