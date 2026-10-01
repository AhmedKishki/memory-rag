"""Instructions sent to any client that connects to this server.

They are written to one standard: a reader who has never seen this server should
finish them knowing what each tool does, which one to call, and when. The order
of the sections is the order of the work — when the memory applies, what to do
before recording anything, which memory a rule belongs in, then the tools, then
what comes back.
"""

from __future__ import annotations

__all__ = ["SERVER_INSTRUCTIONS"]

SERVER_INSTRUCTIONS = """\
This server remembers things on behalf of one project and one account, and it
has four tools: record, recall, forget, and a handoff for the next session.

Two memories, and the difference is who can read them:

- local  this project's memory. It lives in the repository under .memory-rag and
        travels with the project. A convention here, a decision about this code, a
        path in this repository, what this project is for.
- global the account's memory. It is shared by every project on this machine.
        What is true of the user, and an instruction they mean everywhere.

The server is bound to one project when it starts, so nothing takes a project
argument and the account's memory has no user identifier.

WHEN SOMETHING BELONGS HERE

A statement is something meant to hold beyond this reply: a rule, a guiding
principle, a decision, a correction, a preference, a path, what a project is for.
The test is whether the next request in three weeks would still be governed by
it. "Always cite the commit that introduced a change" is one. "Rename this file"
is not: it is finished when it is answered, and nothing later depends on it.

Write what holds, not what happened. A statement carries no date and no time,
because the memory dates every statement itself and a recall reports that date.
The server refuses a statement carrying one: a statement that says when something
was approved is one row per session saying that the work was done, rather than
one approval. A year on its own is fine — "(HUP 1971)" says which edition is meant
and that still holds next year.

So "Section 4 T03 is settled by the author's approval" is not a statement, because
in a month it is a row about a session rather than a fact about the draft. What
holds is "section 4 keeps Marx's categories as supporting vocabulary under
Gidwani's framework". Record that, and let the date be the record's own.

RECALL BEFORE YOU RECORD

Whenever something is about to be recorded, recall it first, with a few words
covering the same thing. Three outcomes, three actions:

  nothing relevant comes back      record it
  something comes back that says   record nothing; it is already known, and a
    the same thing                 second copy makes the answer worse
  something comes back that        forget that statement, then record this one
    contradicts it

Forgetting is for a statement that has stopped being true, and a contradicted
statement is the usual reason. It is also for one that is merely wrong.

WHICH MEMORY A RULE BELONGS IN

Two destinations, and nothing else to choose between:

  local   this project. It lives in the repository and travels with it.
  global  across projects. It is the account's, shared by every project on this
          machine.

Which one a statement belongs in is read off what the user is actually asking
for and how far they mean it to reach — a fact about this repository is local
whatever it is called, and a preference about how they want to be spoken to, or
a rule about their own work rather than about this code, belongs to the account.
The judgement is semantic, and it is usually made on the substance of what is
being remembered rather than on anything said about where it should go: most
prompts carry no marker of their own, so the absence of one is not a reason to
file local.

`scope` defaults to `local`, and a default is not an answer: pass `global` when
the statement is meant to hold beyond this project. When a rule is genuinely
both, record it in the project and say in your answer that it applies more
widely.

THE TOOLS

record_memory(content, kind, scope)
    Records one statement. `scope` is "local" for this project or "global" for
    across projects, and defaults to "local". Answers with the kind it was filed
    under and the memory it went to.

recall_memory(query, kind, limit)
    Returns what is remembered that matches `query`, searching both memories and
    ranking the results by how well they match, so the best statement wins
    whichever memory it is in. Every statement names its scope. `kind` narrows
    the answer to one category.

forget_memory(text, scope)
    Removes the one statement whose text is exactly `text`. `scope` limits the
    search to one memory, and without it both are searched.

record_handoff(content)
    Records this session's handoff for the next session, and removes the previous
    one. Always this project's memory.

Every answer that returns statements gives each one with its scope, its kind, its
text, the date it was added where it has one, and how many times it has been
recalled. A statement that has never come up and one that comes up constantly are
not equally worth keeping.

KINDS

A kind is one word in block letters, and it exists so a later recall can ask for
one category. The usual ones:

  RULE        an instruction that was given, or a standing fact that holds
  PLAN        what the project is meant to be, or what it is achieving
  PREFERENCE  what the user likes or wants
  CORRECTION  something to stop doing

Any other word is accepted and nothing here interprets it, so a kind of your own
is filed under that word rather than under ITEM. The same word should be used for
the same kind of thing. ITEM is the kind a statement gets when none is named, and
it is a kind like any other: a recall filtered by ITEM finds those statements.

A kind is a column of the record rather than part of a statement's words, so a
query that names a kind finds nothing. Pass it as `kind` instead.

ENDING A SESSION

A session another session will continue is ended with record_handoff: what is
done, what is in flight, and what to do first. It replaces the previous handoff,
so a project holds one at a time and a later session never has to work out
whether it is reading today's or last week's.

Work that was finished belongs there and not in a statement. "The T03 amendments
were approved" is a moment, and a statement of it is a row that goes stale while
the memory around it stays true. The handoff says it, keeps it until the session
after this one, and then lets it go.

WHAT ELSE AN ANSWER TELLS YOU

An answer carries a field only when the field has news. Every statement comes
back with its words, its kind, the memory it is in, and how often that memory
has handed it over; a statement with a date carries the date. Beyond that:

  hint                 present only when nothing matched, and says what to try
  truncated            present only when more matched than you were shown
  units_pending        statements findable by words whose vector is not ready
  semantic_available   present only as false, when only the words were searched,
                       so only the exact words would have found anything
  superseded_removed   a file an earlier version left was read for anything it
                       held that the memory did not have, and then removed
  collapsed_repetitions a statement that said the same thing, and which test
                       caught it
  vectors_removed       on a write, vectors of statements that no longer exist,
                       which are dropped so the next settle re-embeds what is left

Nothing else is in an answer: no score, no timing, no position in the document,
and no count a caller could make for itself. An answer is what was remembered,
ordered.

Nothing here is a transcript. What a recall returns is the record of what was
remembered, so quote it as it was recorded and let the user decide which
statement matters when two of them disagree.
"""
