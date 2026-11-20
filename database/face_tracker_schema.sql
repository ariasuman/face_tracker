
-- Face Tracker Database

CREATE DATABASE IF NOT EXISTS face_tracker;

USE face_tracker;

-- Table 1: Store registered faces
CREATE TABLE IF NOT EXISTS faces (
    face_id VARCHAR(50) PRIMARY KEY,
    embedding BLOB NOT NULL,
    first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_seen DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Table 2: Store visitor visits
CREATE TABLE IF NOT EXISTS visits (
    visit_id INT AUTO_INCREMENT PRIMARY KEY,
    face_id VARCHAR(50) NOT NULL,
    entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    exit_time DATETIME NULL,
    status ENUM('ACTIVE', 'COMPLETED') DEFAULT 'ACTIVE',
    FOREIGN KEY (face_id) REFERENCES faces(face_id)
);

-- Table 3: Store visitor events
CREATE TABLE IF NOT EXISTS events (
    event_id INT AUTO_INCREMENT PRIMARY KEY,
    visit_id INT NULL,
    face_id VARCHAR(50) NOT NULL,
    event_type ENUM(
        'REGISTRATION',
        'RECOGNITION',
        'ENTRY',
        'EXIT'
    ) NOT NULL,
    event_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    image_path VARCHAR(255),
    FOREIGN KEY (face_id) REFERENCES faces(face_id),
    FOREIGN KEY (visit_id) REFERENCES visits(visit_id)
);
