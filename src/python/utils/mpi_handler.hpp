/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2026 Simons Foundation & The CoQuí developer team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==========================================================================
 */


#ifndef MPI_HANDLER_HPP
#define MPI_HANDLER_HPP

#include "utilities/mpi_context.h"

namespace coqui_py {

  /**
   * @brief mpi handler class
   *
   * The MpiHandler class encapsulates the state of a MPI environment used by CoQui.
   * It manages key information such as the total number of processors, node distribution,
   * and provides access to global, internode, and intranode communicators.
   *
   * This class also offers a minimal interface for performing basic MPI operations.
   * It must be constructed and passed to any CoQuí routines that involve MPI parallelization.
   * Even in serial mode (i.e., when using a single process), this class is required to ensure
   * a consistent interface across all workflows.
   */
  class MpiHandler {
  public:
    using mpi_context_t = utils::mpi_context_t<mpi3::communicator>;
  public:
    MpiHandler(): _mpi(std::make_shared<mpi_context_t>(utils::make_mpi_context()) ) {}
    C2PY_IGNORE
    MpiHandler(std::shared_ptr<mpi_context_t> mpi): _mpi(std::move(mpi)) {}

    ~MpiHandler() = default;
    MpiHandler(MpiHandler const& other) = delete;
    MpiHandler(MpiHandler &&) = default;
    MpiHandler& operator=(MpiHandler const& other) = delete;
    MpiHandler& operator=(MpiHandler &&) = default;

    bool operator==(const MpiHandler& other) const {
      return _mpi == other._mpi;
    }

    /**
     * @return true if the current process is the root, false otherwise.
     */
    auto root() const { return _mpi->comm.root(); }
    /**
     * @return the rank of the current process in the global communicator.
     */
    auto comm_rank() const { return _mpi->comm.rank(); }
    /**
     * @return the size of the global communicator, i.e., the total number of processes.
     */
    auto comm_size() const { return _mpi->comm.size(); }
    /**
     * @return the rank of the current process in the internode communicator.
     */
    auto internode_rank() const { return _mpi->internode_comm.rank(); }
    /**
     * @return the size of the internode communicator, i.e., the number of nodes.
     */
    auto internode_size() const { return _mpi->internode_comm.size(); }
    /**
     * @return the rank of the current process in the intranode communicator.
     */
    auto intranode_rank() const { return _mpi->node_comm.rank(); }
    /**
     * @return the size of the intranode communicator, i.e., the number of processes within a node.
     */
    auto intranode_size() const { return _mpi->node_comm.size(); }

    /**
     * MPI barrier for the global communicator.
     */
    void barrier() const { _mpi->comm.barrier(); }
    /**
     * MPI barrier for the internode communicator.
     */
    void internode_barrier() const { _mpi->internode_comm.barrier(); }
    /**
     * MPI barrier for the intranode communicator.
     */
    void intranode_barrier() const { _mpi->node_comm.barrier(); }

    C2PY_IGNORE
    auto& get_mpi() { return _mpi; }

    friend std::ostream& operator<<(std::ostream& out, const MpiHandler& handler) {
      out << "CoQuí MPI state\n"
          << "-----------------\n"
          << "  Global communicator: rank " << handler.comm_rank()+1
          << " / " << handler.comm_size() << '\n'
          << "  Internode communicator: rank " << handler.internode_rank()+1
          << " / " << handler.internode_size() << '\n'
          << "  Intranode communicator: rank " << handler.intranode_rank()+1
          << " / " << handler.intranode_size();
      return out;
    }

  private:
    std::shared_ptr<mpi_context_t> _mpi;

  };

} // coqui_py

#endif
