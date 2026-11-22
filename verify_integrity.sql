-- Run after a LIVE test against a TEST database. Every query should return 0 rows
-- (except the last two, which are summaries).

-- 1. ACTIVE visits left open (should be none after a clean run)
SELECT visit_id, face_id, entry_time FROM visits WHERE status = 'ACTIVE';

-- 2. Visits that do not have exactly one ENTRY
SELECT visit_id, SUM(event_type='ENTRY') AS entries
FROM events GROUP BY visit_id HAVING entries <> 1;

-- 3. EXIT events without a COMPLETED visit, or more than one EXIT per visit
SELECT visit_id, SUM(event_type='EXIT') AS exits
FROM events GROUP BY visit_id
HAVING exits > 1;

SELECT v.visit_id FROM visits v
LEFT JOIN events e ON e.visit_id = v.visit_id AND e.event_type = 'EXIT'
WHERE v.status = 'COMPLETED' AND e.visit_id IS NULL;

-- 4. Same face with more than one ACTIVE visit
SELECT face_id, COUNT(*) FROM visits WHERE status='ACTIVE' GROUP BY face_id HAVING COUNT(*) > 1;

-- 5. EXIT earlier than ENTRY
SELECT visit_id FROM visits WHERE exit_time IS NOT NULL AND exit_time < entry_time;

-- 6. Events whose image file you can't find on disk: compare image_path values manually
SELECT event_type, COUNT(*) FROM events GROUP BY event_type;

-- 7. Unique visitors
SELECT COUNT(*) AS unique_visitors FROM faces;
