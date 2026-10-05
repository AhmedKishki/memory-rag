"""Instructions sent to any client that connects to this server.

They are written to one standard: a reader who has never seen this server should
finish them knowing which tool to call for what, and when. The order of the sections
is the order of the work — when the memory applies, what to do before recording
anything, which kind a statement belongs in, then the tools, then what comes back.

The kind list is written from :data:`memory_rag.store.KINDS` rather than typed here,
so the sentence an agent reads about a kind and the behaviour that kind has are one
fact in one place rather than two that can drift apart.
"""

from __future__ import annotations

import textwrap

from .store import KINDS, SCOPE_EITHER, SCOPE_GLOBAL

__all__ = ["SERVER_INSTRUCTIONS"]


def _kinds_section() -> str:
    """Return the paragraph describing every kind, written from the registry.

    Each entry says what goes in that kind and, where the kind decides it, which
    memory it lands in and whether the next one of that kind replaces it. Those are
    the three facts an agent cannot infer from a tool name, and they are the ones
    that decide whether a statement is filed where it belongs. A kind's own summary
    says only what goes in it, so the behaviour is stated once and here rather than
    twice in the same sentence.
    """

    lines: list[str] = []
    for kind in KINDS:
        note = ""
        if kind.scope == SCOPE_GLOBAL:
            note = (
                " It always goes in the account's memory, because what is true of the "
                "user is true of them in every project, so the tool takes no memory "
                "argument."
            )
        elif kind.scope != SCOPE_EITHER:
            note = (
                " It always goes in this project's memory, because a handoff is this "
                "project's own state, so the tool takes no memory argument."
            )
        if kind.replaces:
            note += (
                " It holds one statement rather than a list: the next handoff removes "
                "this one."
            )
        wrapped = textwrap.wrap(
            f"{kind.tool} — {kind.summary}{note}",
            width=76,
            initial_indent="  ",
            subsequent_indent="      ",
        )
        lines.append("\n".join(wrapped))
    return "\n".join(lines)


def _tools_section() -> str:
    """Return the tool list, written from the registry so it cannot fall behind it."""

    recorders = "\n".join(
        f"  record_memory_{kind.tool}(content"
        + (", scope)" if kind.scope == SCOPE_EITHER else ")")
        for kind in KINDS
    )
    recallers = ", ".join(f"recall_memory_{kind.tool}" for kind in KINDS)
    return f"""THE TOOLS

Ten recording tools, one per kind. Each takes the one statement to remember, and the
kind is the tool rather than an argument:

{recorders}

Every recorder except `record_memory_personality` and `record_memory_preference`
takes `scope`: "local" for this project, inside the repository, and "global" for the
account's memory, shared by every project on this machine. Read it off what the user
is actually asking for and how far they mean it to reach — a fact about this
repository is local whatever it is called. Prompts rarely carry a marker of their
own, so judge the substance and not the wording.

Two recall tools per kind answer a question from that kind alone:

  {recallers}

and one more answers across every kind at once:

  recall_memory(query, limit)

Reach for a `recall_memory_<kind>` tool when you know which kind you want, and for
`recall_memory` when you are asking about a subject rather than a category. It is the
one to try first, because an agent asking what is remembered rarely knows which kind
filed it.

forget_memory(text, scope)
    Removes the one statement whose text is exactly `text`, whatever kind it was
    filed under. `scope` limits the search to one memory, and without it both are
    searched. When the text matches more than one statement, nothing is removed and
    the answer names them.

Every answer that returns statements gives each one with its scope, its kind, its
text, the date it was added where it has one, and how many times it has been
recalled. A statement that has never come up and one that comes up constantly are
not equally worth keeping."""


SERVER_INSTRUCTIONS = f"""\
This server remembers things on behalf of one project and one account, by kind.
What a statement means decides which tool records it, so the kind is named by the
tool rather than passed to one.

Two memories, and the difference is who can read them:

- local  this project's memory. It lives in the repository under .memory-rag and
        travels with the project. A convention here, a decision about this code, a
        path in this repository, what this project is for.
- global the account's memory. It is shared by every project on this machine.
        What is true of the user, and an instruction they mean everywhere.

The server is bound to one project when it starts, so nothing takes a project
argument and the account's memory has no user identifier.

WHEN SOMETHING BELONGS HERE

A statement is something meant to hold beyond this reply: a rule, a guideline, a
decision, a correction, a preference, a limit, what a project is for, what a user is
like. The test is whether the next request in three weeks would still be governed by
it. "Always cite the commit that introduced a change" is one. "Rename this file" is
not: it is finished when it is answered, and nothing later depends on it.

Write what holds, not what happened. A statement carries no date and no time,
because the memory dates every statement itself and a recall reports that date.
The server refuses a statement carrying one: a statement that says when something
was approved is one row per session saying that the work was done, rather than one
approval. A year on its own is fine — "(HUP 1971)" says which edition is meant and
that still holds next year.

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

WHICH KIND A STATEMENT BELONGS IN

There is no kind argument and no default, so choose the kind first and let the
statement follow from it:

{_kinds_section()}

There is no other kind. A word that is not one of these is refused with the list
beside it, because a kind nothing interprets is a category a later recall cannot
ask for.

{_tools_section()}

ENDING A SESSION

A session another session will continue is ended with record_memory_handoff: what is
done, what is in flight, and what to do first. It replaces the previous handoff, so a
project holds one at a time and a later session never has to work out whether it is
reading today's or last week's.

Work that was finished belongs there and not in a statement. "The T03 amendments
were approved" is a moment, and a statement of it is a row that goes stale while the
memory around it stays true. The handoff says it, keeps it until the session after
this one, and then lets it go.

WHAT ELSE AN ANSWER TELLS YOU

An answer carries a field only when the field has news. Every statement comes back
with its words, its kind, the memory it is in, and how often that memory has handed
it over; a statement with a date carries the date. Beyond that:

  hint                 present only when nothing matched, and says what to try
  truncated            present only when more matched than were shown
  units_pending        statements findable by words whose vector is not ready
  semantic_available   present only as false, when only the words were searched,
                       so only the exact words would have found anything
  superseded_removed   a file an earlier version left was read for anything it
                       held that the memory did not have, and then removed
  collapsed_repetitions a statement that said the same thing, and which test
                       caught it
  vectors_removed      on a write, vectors of statements that no longer exist,
                       which are dropped so the next settle re-embeds what is left
  app                  on recall_memory, where the app is and which memories it
                       serves

Nothing else is in an answer: no score, no timing, no position in the document,
and no count a caller could make for itself. An answer is what was remembered,
ordered.

Nothing here is a transcript. What a recall returns is the record of what was
remembered, so quote it as it was recorded and let the user decide which
statement matters when two of them disagree.
"""
