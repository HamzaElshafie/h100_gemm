#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cmath>

#include "utils.h"
#include "runner.cuh"

/**
 * @brief Prints the usage instructions for the program.
 *
 * This function outputs the correct CLI usage for the executable,
 * including the required arguments and an example.
 */
void printUsage() {
    std::cout << "Usage: ./sgemm <implementation> <kernel_ID_number>\n"
              << "  implementation: simon | hopper\n"
              << "  ID:       0, 1, 2, ...\n" // TODO: Print last kernel number for each implementation
              << "Example: ./sgemm simon 0\n";
}

/**
 * @brief Parses the kernel configuration from CLI arguments.
 *
 * @param impl      The implementation name (e.g., "simon" or "hopper").
 * @param kernel_id The kernel variant number.
 * @return KernelConfig The parsed kernel configuration.
 * @throws std::invalid_argument if the kernel_id is invalid.
 */
KernelConfig parseKernelConfig(const std::string& impl, int kernel_id) {
    if (impl == "simon") { // Check kernel validity
        if (kernel_id > 1 || kernel_id < 0) { // (TODO: Update later)
            throw std::invalid_argument("Invalid Simon kernel ID");
        }
        return KernelConfig(KernelType::SIMON, kernel_id);
    } else if (impl == "hopper") {
        if (kernel_id > 1 || kernel_id < 0) { // (TODO: Update later)
            throw std::invalid_argument("Invalid Hopper kernel ID");
        }
        return KernelConfig(KernelType::HOPPER, kernel_id);
    }
}

/**
 * @brief Main entry point for the program.
 *
 * @param argc Number of CLI arguments.
 * @param argv Array of CLI strings.
 * @return int Exit status code.
 */
int main(int argc, char** argv) {
    if (argc != 3) {
        printUsage();
        return -1;
    }
    try {
        std::string impl = argv[1];
        int kernel_id = std::stoi(argv[2]);
        KernelConfig config = parseKernelConfig(impl, kernel_id);
        // TODO: Add kernel launch and further logic here
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}

