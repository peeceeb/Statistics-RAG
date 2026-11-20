"""The RAG pipeline emits events for each query in batches, and an at-least-once message bus means the same event can arrive more than once. Each event carries a Stage and a timestamp (ms). Because of clock skew between services, some events arrive with a timestamp earlier than the query's submitted_at — these are invalid. The platform team also wants to know which stage a query is in right now.
Add two methods to QueryTracker:
2.1) add_events(query_id, events): "events" is a list of (stage, timestamp) pairs. If the query exists, store the valid pairs not already stored and return how many were newly stored; otherwise store nothing and return 0. A pair is a duplicate if the same stage and timestamp is already stored (from an earlier batch or earlier in the same batch). A pair whose timestamp is earlier than the query's submitted_at is ignored.
2.2) get_current_stage(query_id): return the Stage of the stored event with the largest timestamp. If two events share the largest timestamp, return the one with the larger Stage value. Return None if the query has no events or does not exist.
Example (query 40 exists, submitted_at = 100):
add_events(40, [(Stage.RETRIEVED, 150), (Stage.RERANKED, 600)])                       -> 2
add_events(40, [(Stage.RETRIEVED, 150), (Stage.FAILED, 20), (Stage.ANSWERED, 2600)])  -> 1
get_current_stage(40)                                                                  -> Stage.ANSWERED
Explanation: The first call stores both events. In the second call, (RETRIEVED, 150) is already stored and (FAILED, 20) is before submission, so only (ANSWERED, 2600) is new. The latest event is ANSWERED at 2600.
# Add these two methods inside the TestSuite class.
"""
def test_add_events(self):
   """Test add_events."""
   tracker = QueryTracker()
   tracker.add_query(Query(10, 1, 100))
   tracker.add_query(Query(20, 1, 0))
   self.assertEqual(tracker.add_events(10, [(Stage.RETRIEVED, 150), (Stage.RERANKED, 400)]), 2)
   # (RETRIEVED, 150) already stored -> only (ANSWERED, 2100) is new.
   self.assertEqual(tracker.add_events(10, [(Stage.RETRIEVED, 150), (Stage.ANSWERED, 2100)]), 1)
   # Within-batch duplicate is stored once.
   self.assertEqual(tracker.add_events(20, [(Stage.RETRIEVED, 30), (Stage.RETRIEVED, 30)]), 1)
   # Event before the query was submitted (clock skew) is rejected.
   self.assertEqual(tracker.add_events(10, [(Stage.FAILED, 50)]), 0)
   # Unknown query: nothing stored.
   self.assertEqual(tracker.add_events(999, [(Stage.RETRIEVED, 1)]), 0)
 
def test_get_current_stage(self):
   """Test get_current_stage."""
   tracker = QueryTracker()
   tracker.add_query(Query(10, 1, 0))
   tracker.add_query(Query(20, 1, 0))
   tracker.add_query(Query(30, 2, 0))   # never processed
   # Events arrive out of order: the latest by timestamp is ANSWERED at 4000.
   tracker.add_events(10, [(Stage.ANSWERED, 4000), (Stage.RETRIEVED, 200), (Stage.RERANKED, 900)])
    self.assertEqual(tracker.get_current_stage(10), Stage.ANSWERED)
   # A later batch can change the current stage.
   tracker.add_events(10, [(Stage.FAILED, 9000)])
    self.assertEqual(tracker.get_current_stage(10), Stage.FAILED)
   # Tie on the latest timestamp -> the stage with the larger value wins.
   tracker.add_events(20, [(Stage.RERANKED, 500), (Stage.RETRIEVED, 500), (Stage.RETRIEVED, 100)])
    self.assertEqual(tracker.get_current_stage(20), Stage.RERANKED)
    self.assertIsNone(tracker.get_current_stage(30))    # exists but no events
    self.assertIsNone(tracker.get_current_stage(999))   # no such query

"""
Clarifications
Events can arrive out of order; the current stage is decided by timestamp, not insertion order.
Two events that differ in either stage or timestamp are both kept.
An event with timestamp equal to submitted_at is valid.
Pitfalls
Returning the last event stored instead of the event with the largest timestamp.
Ignoring the tie-break rule when two events share the latest timestamp.
Calling max() on an empty collection for a query with no events.
Storing events for a query that does not exist, or storing pre-submission (clock-skew) events.
Counting a rejected or duplicate event in the return value.
"""