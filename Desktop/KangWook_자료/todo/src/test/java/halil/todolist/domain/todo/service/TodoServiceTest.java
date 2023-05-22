package halil.todolist.domain.todo.service;

import halil.todolist.domain.member.entity.Member;
import halil.todolist.domain.member.repository.MemberRepository;
import halil.todolist.domain.todo.entity.Todo;
import halil.todolist.domain.todo.repository.TodoRepository;
import org.apache.catalina.connector.Request;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import javax.servlet.http.HttpServletRequest;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
class TodoServiceTest {

    @Autowired
    MemberRepository memberRepository;

    @Autowired
    TodoRepository todoRepository;

    @Test
    void addTodo() {
        Member memberByEmail = memberRepository.findMemberByEmail("2@naver.com");

        Todo build = Todo.builder()
                .member(memberByEmail)
                .text("asd")
                //.status("YEs")
                .build();

        todoRepository.save(build);
    }

}