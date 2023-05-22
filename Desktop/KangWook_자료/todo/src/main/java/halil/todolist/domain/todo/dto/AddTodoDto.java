package halil.todolist.domain.todo.dto;

import halil.todolist.domain.todo.entity.Status;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.springframework.data.annotation.CreatedDate;
import org.springframework.format.annotation.DateTimeFormat;

import javax.persistence.Column;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

@Getter
@Setter
@NoArgsConstructor
public class AddTodoDto {

    @Column(nullable = false)
    private String text;

    private Status status;

    @CreatedDate
    @Column(updatable = false)
    private String createDateTime = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy.MM.dd HH:mm:ss"));
}
