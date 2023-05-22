package halil.todolist.security.manager;

import halil.todolist.domain.member.entity.Member;
import halil.todolist.domain.member.repository.MemberRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Component;

import java.util.Optional;

@Component
@RequiredArgsConstructor
public class CustomAuthenticationManager implements AuthenticationManager {

    private final MemberRepository memberRepository;
    private final PasswordEncoder passwordEncoder;

    @Override
    public Authentication authenticate(Authentication authentication) throws AuthenticationException {
        Optional<Member> member = memberRepository.findByEmail(authentication.getName());
        if (!passwordEncoder.matches(
                authentication.getCredentials().toString(),
                member.get().getPassword())) {
            throw new BadCredentialsException("잘못된 비밀번호");
        }
        return new UsernamePasswordAuthenticationToken(authentication.getName(), member.get().getPassword());
    }
}
