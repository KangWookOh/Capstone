package halil.todolist.domain.todo.repository;

import halil.todolist.domain.todo.entity.Todo;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;

public interface TodoRepository extends JpaRepository<Todo, Long> {

    @Query("select t " +
            " from Todo t" +
            " where t.member.id =:id")
    List<Todo> findTodobyMember(@Param("id") Long id);
}

