#include "bearing/core/ConcurrentQueue.hpp"

#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <thread>

int main() {
    // ── 用例 1：无界队列基本读写 + stop() 唤醒消费者 ──────────────────────
    {
        bearing::ConcurrentQueue<int> queue;

        queue.push(42);
        int value = 0;
        if (!queue.pop(value) || value != 42) {
            std::cerr << "queue did not return the pushed value" << std::endl;
            return EXIT_FAILURE;
        }

        auto waiting_consumer = std::async(std::launch::async, [&queue]() {
            int stopped_value = 0;
            return queue.pop(stopped_value);
        });

        queue.stop();
        auto state = waiting_consumer.wait_for(std::chrono::seconds(1));
        if (state != std::future_status::ready || waiting_consumer.get()) {
            std::cerr << "stop did not wake the blocked consumer" << std::endl;
            return EXIT_FAILURE;
        }
    }

    // ── 用例 2：有界队列（容量 2）—— 第 3 次 push 被 pop 解除阻塞 ─────────
    // 验证语义：有界队列满时 push() 阻塞，pop() 取走一个元素后 push() 继续。
    {
        bearing::ConcurrentQueue<int> bounded_queue(2);

        // 填满队列（容量 2），这两次 push 不应阻塞
        bounded_queue.push(1);
        bounded_queue.push(2);

        // 第 3 次 push 应阻塞（队列已满）；用 std::async 在独立线程里推送，
        // 然后用 wait_for 设置 2s 硬超时，防止测试因死锁永久挂起。
        auto blocked_push = std::async(std::launch::async, [&bounded_queue]() {
            return bounded_queue.push(3);
        });

        // 给生产者线程 10ms 时间进入 condition_.wait()（队列已满，必然阻塞）
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // pop 取走一个元素，应唤醒被阻塞的生产者
        int popped = 0;
        if (!bounded_queue.pop(popped) || popped != 1) {
            std::cerr << "bounded queue: pop returned unexpected value" << std::endl;
            return EXIT_FAILURE;
        }

        // 等待生产者完成（最多 2s）
        if (blocked_push.wait_for(std::chrono::seconds(2)) != std::future_status::ready) {
            std::cerr << "bounded queue: blocked push did not unblock after pop" << std::endl;
            return EXIT_FAILURE;
        }
        if (!blocked_push.get()) {
            std::cerr << "bounded queue: push returned false unexpectedly" << std::endl;
            return EXIT_FAILURE;
        }
    }

    // ── 用例 3：有界队列 —— stop() 唤醒被阻塞的生产者，push() 返回 false ──
    // 验证语义：stop() 后阻塞中的 push() 应立即返回 false（不再入队）。
    {
        bearing::ConcurrentQueue<int> bounded_queue(2);

        bounded_queue.push(10);
        bounded_queue.push(20);

        // 第 3 次 push 阻塞（队列满）
        auto blocked_push = std::async(std::launch::async, [&bounded_queue]() {
            return bounded_queue.push(30);
        });

        // 给生产者线程 10ms 时间进入 condition_.wait()
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        // stop() 应唤醒被阻塞的生产者，使其返回 false
        bounded_queue.stop();

        if (blocked_push.wait_for(std::chrono::seconds(2)) != std::future_status::ready) {
            std::cerr << "bounded queue: stop() did not wake blocked producer" << std::endl;
            return EXIT_FAILURE;
        }
        if (blocked_push.get()) {
            std::cerr << "bounded queue: push returned true after stop()" << std::endl;
            return EXIT_FAILURE;
        }
    }

    return EXIT_SUCCESS;
}
