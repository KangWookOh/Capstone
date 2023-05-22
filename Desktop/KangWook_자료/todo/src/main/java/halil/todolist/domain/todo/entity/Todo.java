package halil.todolist.domain.todo.entity;

import halil.todolist.domain.member.entity.Member;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

import javax.persistence.*;
import java.time.LocalDateTime;

@Entity
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
public class Todo {

    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String text;

    @Enumerated(EnumType.STRING)
    private Status status;

    private String createDateTime;

    @ManyToOne
    private Member member;

    public void updateTodo(Status status) {
        this.status = status;
    }

    @Builder
    public Todo(String text, Status status, Member member, String createDateTime) {
        this.text = text;
        this.status = status;
        this.member = member;
        this.createDateTime = createDateTime;
    }
}
