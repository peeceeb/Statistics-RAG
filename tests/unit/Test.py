"""
We are building the backend that tracks questions sent to the bank's RAG assistant. 
When a user submits a question, the service creates a Query for that user's team. One team submits many queries.
Definitions:
A "Query" is one user question. It has a query_id, a team_id, and a submitted_at time (milliseconds since the start of the day).
A "Stage" is a pipeline event recorded for the query: RETRIEVED, RERANKED, ANSWERED or FAILED. (Used from Question 2 onward.)
A "QueryTracker" is the class used to manage all queries for the day.

What the code does today: 
QueryTracker keeps every Query in a list. 
--add_query() registers a query, 
--get_query() is meant to return the query with a given ID (or None), and 
--get_team_queries() returns the IDs of all queries from one team, sorted ascending.

Two tasks:
Read through and understand the code below. 
Take as much time as you need, and feel free to run it.
One of the tests is not passing due to a bug. 
Make the necessary change to QueryTracker to fix it.
"""

"""
Query-----> Query(query_id, team_id, submitted_at)
Stage-----> Stage(Enum): RETRIEVED, RERANKED, ANSWERED, FAILED
QueryTracker-----> QueryTracker: add_query(), get_query(), get_team_queries()
"""


import unittest
from enum import Enum
 
 
class Stage(Enum):
   RETRIEVED = 1
   RERANKED = 2
   ANSWERED = 3
   FAILED = 4
output = Stage.FAILED.value

print("Stage_Retrieved =",output)
 
class Query:
   """Data about a single user question sent to the RAG assistant."""
 
   def __init__(self, query_id, team_id, submitted_at):
       self.query_id = query_id
       self.team_id = team_id
       self.submitted_at = submitted_at  # milliseconds since the start of the day
 
   def __eq__(self, other):
       if not isinstance(other, self.__class__):
           return False
       return (
           self.query_id == other.query_id
           and self.team_id == other.team_id
           and self.submitted_at == other.submitted_at
       )
 
   def __str__(self):
       return f"Query {self.query_id} (team {self.team_id}) submitted={self.submitted_at}"


Test_Query = Query(1, 101, 300)
print("Query ID:", Test_Query.query_id) 
print("Team ID:", Test_Query.team_id)
print("Submitted at:", Test_Query.submitted_at)

 
 
class QueryTracker:
   """Tracks RAG queries for the day, and methods the platform team uses to query them."""
 
   def __init__(self):
       self.queries = []
 
   def add_query(self, query):
       """Register a query for the current day."""
       self.queries.append(query)
 
   def get_query(self, team_id):
       """Returns the query with the given ID, or None if there is no such query."""
       for query in self.queries:
           if query.team_id == team_id:
               return query
       return None
 
   def get_team_queries(self, team_id):
       """Returns the IDs of all queries from the given team, sorted in ascending order."""
       result = []
       for query in self.queries:
           if query.team_id == team_id:
               result.append(query.query_id)
       result.sort()
       return result

Test_Tracker = QueryTracker()
Test_Tracker.add_query(Query(2, 201, 300))
Test_Tracker.add_query(Query(3, 201, 400))

print(Test_Tracker.get_query(201))  # Should return the first query for team 201
print(Test_Tracker.get_team_queries(201))  # Should return the IDs of all queries for team 201


class TestSuite(unittest.TestCase):
   def test_query(self):
       """Test basic Query construction."""
       q = Query(1, 12, 900)
       self.assertEqual(q.query_id, 1)
       self.assertEqual(q.team_id, 12)
       self.assertEqual(q.submitted_at, 900)
       print("Test_query",q)
 
   def test_get_query(self):
       tracker = QueryTracker()
       first = Query(1, 7, 0)
       second = Query(7, 1, 100)
       third = Query(3, 9, 200)
       tracker.add_query(first)
       tracker.add_query(second)
       tracker.add_query(third)
       # Lookup is by team ID - team 1 owns 'second', not 'first'.
       print(self.assertEqual(tracker.get_query(7), first))
       print(self.assertEqual(tracker.get_query(1), second))
       print(self.assertEqual(tracker.get_query(9), third))
       # Unknown team returns None.
       #self.assertIsNone(tracker.get_query(42))
       # An empty tracker returns None.
       #self.assertIsNone(QueryTracker().get_query(1))

Test_Suite = TestSuite()
print(Test_Suite.test_query())
print(Test_Suite.test_get_query())



"""The RAG pipeline emits events for each query in batches, and an at-least-once message bus means the same event can arrive more than once. 
Each event carries a Stage and a timestamp (ms). 
Because of clock skew between services, some events arrive with a timestamp earlier than the query's submitted_at — these are invalid. 
The platform team also wants to know which stage a query is in right now.

Add two methods to QueryTracker:

2.1) add_events(query_id, events): "events" is a list of (stage, timestamp) pairs. 
If the query exists, store the valid pairs not already stored and return how many were newly stored; 
otherwise store nothing and return 0. A pair is a duplicate if the same stage and timestamp is already stored (from an earlier batch or earlier in the same batch). 
A pair whose timestamp is earlier than the query's submitted_at is ignored.

2.2) get_current_stage(query_id): return the Stage of the stored event with the largest timestamp. 
If two events share the largest timestamp, return the one with the larger Stage value. 
Return None if the query has no events or does not exist.

Example (query 40 exists, submitted_at = 100):
add_events(40, [(Stage.RETRIEVED, 150), (Stage.RERANKED, 600)])                       -> 2
add_events(40, [(Stage.RETRIEVED, 150), (Stage.FAILED, 20), (Stage.ANSWERED, 2600)])  -> 1
get_current_stage(40)                                                                  -> Stage.ANSWERED

Explanation: The first call stores both events. In the second call, (RETRIEVED, 150) is already stored and (FAILED, 20) is before submission, so only (ANSWERED, 2600) is new. The latest event is ANSWERED at 2600.
# Add these two methods inside the TestSuite class.
"""

def add_events(query_id, events):
    if query_id in events:
        pass
    else:
        return None



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