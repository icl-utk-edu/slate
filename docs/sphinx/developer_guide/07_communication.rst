Message Passing Communication
=============================

Communication in SLATE relies on explicit dataflow information. When a tile will be needed for computation, it is broadcast to all the processes where it is required. Rather than explicitly listing MPI ranks, the broadcast is expressed in terms of the destination tiles to be updated. ``tileBcast`` takes a tile's :math:`(i, j)` indices and a sub-matrix that the tile will update; the tile is sent to all processes owning that sub-matrix.

To optimize communication, ``listBcast`` aggregates a list of these tile broadcasts and pipelines the MPI and CPU-to-accelerator communication. As the set of processes involved is dynamically determined from the sub-matrix, using an MPI broadcast would require setting up a new MPI communicator, which is an expensive global blocking operation. Instead, SLATE uses point-to-point MPI communication in a hypercube tree fashion to broadcast the data.
